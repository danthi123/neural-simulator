"""Generalization v7 gate: v6 silent recall + intrinsic threshold homeostasis.

v6 calibration was GO but development was a NO-GO on ONE of twenty frozen
components, ``weakest_source_margin_strictly_improved`` (seed 654): the fixed
symmetric GABA-A competition lifted the second-strongest source, not the
weakest, so ``min(M) > min(L)`` failed by exactly 0.0
(2026-08-06-source-monitor-coresidency-v6-development-NO-GO-...).  The named
surpass is a NEW METHOD for the SAME frozen criterion: add Turrigiano intrinsic
threshold homeostasis (activity-dependent excitability) on the source-memory
populations so the least-active source is up-regulated and the strict-improvement
criterion holds on every development seed rather than by seed luck.

v7 = v6 UNCHANGED (silent-by-construction settle-to-quiescence recall, fixed
local fast-spiking GABA-A competition, and the identical bounded-loss max-min
acceptance rule with all twenty frozen components + thresholds) PLUS the shipped
region-scoped intrinsic homeostasis used by v3: the source-memory regions opt
into ``enable_homeostasis`` so the substrate maintains a per-neuron firing-rate
EMA and lowers the firing threshold of an under-firing population toward its
floor (``sim/kernels.py::fused_homeostasis_update``; Turrigiano, activity-
dependent intrinsic plasticity).  After learning, the intact network settles
under balanced, learning-off rehearsal with homeostatic updates ON, which
lowers the weakest source's spike threshold; recall then reads the frozen
network unchanged in kind.  The homeostasis is the substrate's own intrinsic
mechanism, not a host reimplementation, and nothing else changes.

v3 alone was a NO-GO because at the CANONICAL operating point the threshold
barely moved over one EMA time constant (~0.05 mV / 5000 steps -> effectively
inert), so the mechanism made no redistributive difference.  v7 gives the SAME
shipped mechanism enough authority to reach an activity-dependent steady state
(the operating point is characterized and frozen on the CALIBRATION seeds, never
on the held development/held-out seeds), so the up-regulation is real rather
than negligible.  The acceptance criteria and thresholds are NOT loosened.

Seed 648 is reserved for non-scientific construction/dynamics smoke.  Only
calibration seeds 650 and 651 are open for operating-point characterization.
Development opens 652/653/654; held-out opens 655/656/657 ONLY after development
records an earned GO here (``validate_phase_seed`` seals it otherwise).  Only the
seed partition advances; the circuit, thresholds, acceptance rule, and frozen
homeostasis operating point are held constant across development and held-out.
The runner does not choose words, calculate confidence, or decide whether to
respond.
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
    DEVELOPMENT_MIN_ATTRIBUTION_FRACTION,
    DEVELOPMENT_MIN_SOURCE_MARGIN,
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
from research.runners._laneC_source_monitor_coresidency_gate_v5 import (
    SOURCE_COMPETITION_GATE,
    SOURCE_INTERNEURON,
    SourceMonitorConfigV2,
)
from research.runners._laneC_source_monitor_coresidency_gate_v6 import (
    MIN_ATTRIBUTION_FRACTION,
    MIN_SOURCE_MARGIN,
    SourceMonitorCoresidencyGateV6,
)
from sim.backend import to_host
from sim.regions import BrainRegion
from tools.lab import attributable_to
from tools.verdict import UNDEFINED, Verdict


SMOKE_SEED = 648
CALIBRATION_SEEDS = (650, 651)
DEVELOPMENT_SEEDS = (652, 653, 654)
HELD_OUT_SEEDS = (655, 656, 657)

PHASE_SEEDS = {
    "calibration": CALIBRATION_SEEDS,
    "development": DEVELOPMENT_SEEDS,
    "held_out": HELD_OUT_SEEDS,
}

# Aggregate generalization verdicts (written by aggregate_source_monitor_v7_seeds).
GENERALIZATION_DIR = Path("research/findings/raw/source_monitor_v7_generalization")
DEV_VERDICT_PATH = GENERALIZATION_DIR / "development_verdict.json"

# --- Frozen homeostasis operating point -----------------------------------
# The shipped homeostasis integrates ``error * adapt_rate`` into the firing
# threshold every step (sim/kernels.py::fused_homeostasis_update).  v3's NO-GO
# used the canonical adapt_rate=5e-4 over one EMA time constant, which moves the
# threshold ~0.05 mV -- far below the ~mV needed to matter -- so the mechanism
# was inert.  v7 gives the SAME mechanism authority to reach an activity-
# dependent steady state: a faster EMA so the rate estimate converges inside the
# settling window, and an adaptation rate that lets the least-active source's
# threshold approach its floor.  The target rate, threshold bounds, and the
# Turrigiano sign are unchanged.  These values are characterized and FROZEN on
# the calibration seeds (see the finding); the development/held-out seeds never
# touch them.
HOMEOSTASIS_TARGET_RATE = 0.02
HOMEOSTASIS_EMA_ALPHA = 0.01          # tau_ema ~100 steps -> the rate estimate converges within the window
HOMEOSTASIS_ADAPT_RATE = 0.005        # moderate: moves source thresholds ~3 mV (demonstrably active, not saturating)
HOMEOSTASIS_THRESHOLD_MIN = -55.0
HOMEOSTASIS_THRESHOLD_MAX = -30.0
HOMEOSTASIS_SETTLING_STEPS = 5000     # one+ EMA time constant of balanced learning-off rehearsal


@dataclass(frozen=True)
class SourceMonitorConfigV7(SourceMonitorConfigV2):
    """Frozen v2/v6 operating point plus the frozen v7 homeostasis operating point."""

    source_homeostasis_target_rate: float = HOMEOSTASIS_TARGET_RATE
    source_homeostasis_ema_alpha: float = HOMEOSTASIS_EMA_ALPHA
    source_homeostasis_adapt_rate: float = HOMEOSTASIS_ADAPT_RATE
    source_homeostasis_threshold_min: float = HOMEOSTASIS_THRESHOLD_MIN
    source_homeostasis_threshold_max: float = HOMEOSTASIS_THRESHOLD_MAX
    source_homeostasis_settling_steps: int = HOMEOSTASIS_SETTLING_STEPS


class SourceMonitorCoresidencyGateV7(SourceMonitorCoresidencyGateV6):
    """v6 silent-recall circuit with source-memory intrinsic threshold homeostasis.

    The source-memory regions opt into the shipped region-scoped homeostasis
    (``BrainRegion.enable_homeostasis=True`` -> ``cp_homeostasis_neuron_mask``),
    so the substrate maintains their firing-rate EMA and adapts their spike
    thresholds.  Updates are applied only during an explicit ``settle_homeostasis``
    window; during experience and recall the update mask is empty, so recall
    reads a frozen network exactly as v6 does.
    """

    def __init__(
        self,
        *,
        seed: int,
        config: SourceMonitorConfigV7 | Mapping | None = None,
    ):
        c = (
            config
            if isinstance(config, SourceMonitorConfigV7)
            else SourceMonitorConfigV7(**(dict(config) if config else {}))
        )
        super().__init__(seed=seed, config=c)
        if self.bridge.cp_homeostasis_neuron_mask is None:
            raise RuntimeError("source-memory homeostasis mask was not initialized")
        self._source_homeostasis_mask = self.bridge.cp_homeostasis_neuron_mask.copy()
        self.set_local_homeostasis_updates(False)

    @staticmethod
    def _region(name: str, n_neurons: int) -> BrainRegion:
        region = SourceMonitorCoresidencyGateV6._region(name, n_neurons)
        if name in SOURCE_MEMORY.values():
            return replace(region, enable_homeostasis=True)
        return region

    def _build_bridge(self):
        bridge = super()._build_bridge()
        c = self.config
        cfg = bridge.core_config
        cfg.homeostasis_target_rate = float(c.source_homeostasis_target_rate)
        cfg.homeostasis_ema_alpha = float(c.source_homeostasis_ema_alpha)
        cfg.homeostasis_threshold_adapt_rate = float(c.source_homeostasis_adapt_rate)
        cfg.homeostasis_threshold_min = float(c.source_homeostasis_threshold_min)
        cfg.homeostasis_threshold_max = float(c.source_homeostasis_threshold_max)
        return bridge

    def expected_homeostasis_mask(self) -> np.ndarray:
        expected = np.zeros(self.bridge.cp_firing_states.shape[0], dtype=bool)
        for source in SOURCES:
            expected[self._source_memory_indices[source]] = True
        return expected

    def homeostasis_mask(self) -> np.ndarray:
        return np.asarray(to_host(self._source_homeostasis_mask), dtype=bool)

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

    def per_source_threshold_means(self) -> dict[str, float]:
        thresholds = np.asarray(
            to_host(self.bridge.cp_neuron_firing_thresholds), dtype=np.float64
        )
        return {
            source: float(thresholds[self._source_memory_indices[source]].mean())
            for source in SOURCES
        }

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
        """Balanced, learning-off rehearsal that adapts source-memory thresholds.

        Learning is gated off, so learned weights are untouched; only the intrinsic
        firing threshold adapts.  Each source-memory population fires through its
        learned episode routes; the least-active (weakest) source accrues the
        lowest firing-rate EMA and so has its spike threshold lowered the most --
        the Turrigiano up-regulation of the under-active population.
        """

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
            "per_source_threshold_means": self.per_source_threshold_means(),
            "learned_weights_unchanged": bool(
                np.array_equal(before_weights, after_weights)
            ),
        }


def _development_is_go() -> bool:
    """True only if the v7 development aggregate verdict exists and reads GO."""

    try:
        data = json.loads(DEV_VERDICT_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return data.get("phase") == "development" and data.get("verdict") == "GO"


def validate_phase_seed(seed: int, phase: str = "calibration") -> int:
    """Open calibration + development seeds; keep held_out sealed until dev GO."""

    seed = int(seed)
    allowed = PHASE_SEEDS.get(phase)
    if allowed is None:
        raise ValueError(
            f"phase {phase!r} is not a v7 phase; choose from {tuple(PHASE_SEEDS)}"
        )
    if phase == "held_out" and not _development_is_go():
        raise ValueError(
            "held_out seeds are sealed until the development phase records an "
            f"earned GO in {DEV_VERDICT_PATH}; run --phase development first"
        )
    if phase == "calibration" and seed == SMOKE_SEED:
        return seed
    if seed not in allowed:
        raise ValueError(
            f"seed {seed} is not a v7 {phase} seed; allowed={allowed}, "
            f"smoke={SMOKE_SEED}, calibration={CALIBRATION_SEEDS}, "
            f"development={DEVELOPMENT_SEEDS}, held_out={HELD_OUT_SEEDS}"
        )
    return seed


def evaluate_calibration_seed(
    seed: int,
    config: SourceMonitorConfigV7 | None = None,
    *,
    phase: str = "calibration",
) -> dict:
    """Run one seed of ``phase`` and all preregistered controls.

    Identical to v6's evaluator except that, after learning, the intact network
    settles under balanced learning-off rehearsal with homeostatic threshold
    updates ON, up-regulating the weakest source's excitability.  The twenty
    frozen components, thresholds, and acceptance rule are unchanged; the
    homeostasis integrity checks are recorded as preconditions, not as new
    components.
    """

    seed = validate_phase_seed(seed, phase)
    c = config or SourceMonitorConfigV7()
    patterns = make_episode_patterns(seed, 5, c)
    t0 = time.time()

    intact = SourceMonitorCoresidencyGateV7(seed=seed, config=c)
    initial = intact.weight_summary()
    initial_thresholds = intact.source_threshold_vector()
    intact.experience(patterns[0], visual_activity=True)
    intact.experience(patterns[1], auditory_activity=True)
    intact.experience(patterns[2], corollary_discharge=True)
    intact.experience(patterns[3], visual_activity=True, auditory_activity=True)
    learned = intact.weight_summary()
    # v7 addition: intrinsic threshold homeostasis up-regulates the weakest source.
    homeostasis_settling = intact.settle_homeostasis(patterns[:3], updates_enabled=True)
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

    swapped = SourceMonitorCoresidencyGateV7(seed=seed + 10000, config=c)
    swapped.experience(patterns[0], auditory_activity=True)
    swapped.experience(patterns[1], visual_activity=True)
    swapped.settle_homeostasis(patterns[:2], updates_enabled=True)
    swap_zero = swapped.recall(patterns[0])
    swap_one = swapped.recall(patterns[1])

    learning_off = SourceMonitorCoresidencyGateV7(seed=seed + 20000, config=c)
    off_initial = learning_off.weight_summary()
    learning_off.experience(patterns[0], visual_activity=True, learning_enabled=False)
    off_after = learning_off.weight_summary()
    learning_off.settle_homeostasis(patterns[:1], updates_enabled=True)
    off_recall = learning_off.recall(patterns[0])

    margins = {
        "seen": _source_margin(seen, "seen"),
        "heard": _source_margin(heard, "heard"),
        "self_generated": _source_margin(self_generated, "self_generated"),
    }
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
    lesion_margins = {
        source: _source_margin(competition_lesions[source], source) for source in SOURCES
    }
    competition_margin_gains = {
        source: margins[source] - lesion_margins[source] for source in SOURCES
    }
    # Bounded-loss, guard-the-floor, max-min acceptance (P3 functional-role spec).
    bounded_loss = {
        source: max(0.0, lesion_margins[source] - margins[source]) for source in SOURCES
    }
    spendable_surplus = {
        source: max(0.0, lesion_margins[source] - MIN_SOURCE_MARGIN) for source in SOURCES
    }
    bounded_loss_within_surplus = all(
        bounded_loss[source] <= spendable_surplus[source] for source in SOURCES
    )
    weakest_source_strictly_improved = min(margins.values()) > min(lesion_margins.values())

    # --- The twenty FROZEN v6 components, unchanged -----------------------
    components = {
        "learned_routes_start_zero": bool(initial["l1"] == 0.0),
        "experience_changes_synaptic_weights": bool(learned["l1"] > initial["l1"]),
        "seen_source_recalled": bool(_dominant_source(seen) == "seen" and margins["seen"] > 0.0),
        "heard_source_recalled": bool(_dominant_source(heard) == "heard" and margins["heard"] > 0.0),
        "self_source_recalled": bool(
            _dominant_source(self_generated) == "self_generated"
            and margins["self_generated"] > 0.0
        ),
        "all_source_margins_meet_fixed_floor": bool(min(margins.values()) >= MIN_SOURCE_MARGIN),
        "source_swap_follows_afferent_activity": bool(
            _dominant_source(swap_zero) == "heard" and _dominant_source(swap_one) == "seen"
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
            sum(acc_lesion["source_spikes"].values()) >= 0.90 * intact_source_total
            and acc_lesion["acc_spikes"] == 0.0
        ),
        "acc_path_attribution_meets_fixed_floor": bool(
            acc_path_fraction >= MIN_ATTRIBUTION_FRACTION
        ),
        "learning_off_keeps_weights_zero": bool(
            off_initial["l1"] == 0.0 and off_after["l1"] == 0.0
        ),
        "learning_off_has_no_source_recall": bool(sum(off_recall["source_spikes"].values()) == 0.0),
        "unseen_episode_has_no_source_recall": bool(sum(unseen["source_spikes"].values()) == 0.0),
        "source_spikes_reach_apfc_and_acc": bool(
            seen["apfc_source_spikes"]["seen"] > 0.0 and seen["acc_spikes"] > 0.0
        ),
        "competition_circuit_is_active_and_lesionable": bool(
            competition_intact_spikes > 0.0 and competition_lesion_spikes == 0.0
        ),
        "bounded_loss_only_spends_surplus": bool(bounded_loss_within_surplus),
        "weakest_source_margin_strictly_improved": bool(weakest_source_strictly_improved),
        "recall_settle_reaches_quiescence": bool(
            all(
                rec["settle"]["reached_quiescence"]
                for rec in (seen, heard, self_generated, mixed, unseen, off_recall)
            )
        ),
    }

    region_names = {region.name for region in intact.bridge.region_manager.regions()}
    expected_regions = {EPISODE_REGION, ACC_REGION}
    expected_regions.update(SOURCE_AFFERENT.values())
    expected_regions.update(SOURCE_MEMORY.values())
    expected_regions.update(APFC_SOURCE.values())
    expected_regions.update(SOURCE_INTERNEURON.values())
    recall_parameters = list(inspect.signature(SourceMonitorCoresidencyGateV7.recall).parameters)
    mask_is_local = np.array_equal(
        intact.homeostasis_mask(), intact.expected_homeostasis_mask()
    )
    earned = Verdict(f"source-monitor co-residency v7 {phase}")
    earned.require(
        "episode, source, competition, aPFC, and ACC populations share one bridge",
        expected_regions.issubset(region_names),
        expect=True,
    )
    earned.require(
        "recall accepts episode activity without source metadata",
        recall_parameters == ["self", "episode_pattern", "source_path_lesion", "acc_lesion"],
        expect=True,
    )
    earned.require(
        "v7 preserves v1 source input strength",
        c.drive_pA == SourceMonitorConfig().drive_pA
        and c.source_afferent_weight == SourceMonitorConfig().source_afferent_weight,
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
        "homeostasis moved source thresholds and left learned weights untouched",
        homeostasis_settling["threshold_delta_l1"] > 0.0
        and homeostasis_settling["learned_weights_unchanged"],
        expect=True,
    )
    earned.knob(
        "Hebbian learning rate",
        requested=c.hebbian_learning_rate,
        applied=intact.bridge.core_config.hebbian_learning_rate,
    )
    earned.knob(
        "GABA-A rival inhibition",
        requested=c.interneuron_to_rival_weight,
        applied=c.interneuron_to_rival_weight,
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
        after=float(homeostasis_settling["threshold_delta_l1"]),
    )
    earned.disabled(
        "STDP, reward modulation, synaptic scaling, short-term plasticity, and structural plasticity",
        why=(
            "v7 isolates Hebbian source association, fixed local GABA-A competition, "
            "and source-region intrinsic threshold homeostasis"
        ),
    )
    decided = earned.decide(go=all(components.values()), verbose=False)
    phase_label = phase.upper()
    status = (
        "UNDEFINED"
        if decided["status"] == UNDEFINED
        else f"{phase_label}_PASS" if decided["go"] else f"{phase_label}_FAIL"
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
            "intact_source_spikes": intact_source_total,
            "source_lesion_spikes": lesioned_source_total,
            "intact_acc_spikes": float(seen["acc_spikes"]),
            "acc_lesion_spikes": float(acc_lesion["acc_spikes"]),
            "mixed_seen_spikes": float(mixed["source_spikes"]["seen"]),
            "mixed_heard_spikes": float(mixed["source_spikes"]["heard"]),
            "competition_intact_spikes": competition_intact_spikes,
            "competition_lesion_spikes": competition_lesion_spikes,
            "competition_margin_gains": competition_margin_gains,
            "competition_lesion_margins": lesion_margins,
            "bounded_loss": bounded_loss,
            "spendable_surplus": spendable_surplus,
            "minimum_lesion_margin": min(lesion_margins.values()),
            "learning_off_source_spikes": float(sum(off_recall["source_spikes"].values())),
            "learning_off_settle_steps": int(off_recall["settle"]["settle_steps"]),
        },
        "homeostasis": {
            "intact_settling": homeostasis_settling,
            "initial_threshold_mean": float(initial_thresholds.mean()),
            "mask_is_source_local": bool(mask_is_local),
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
            "swap_pattern_zero": swap_zero,
            "swap_pattern_one": swap_one,
            "learning_off": off_recall,
        },
        "weights": {
            "initial": initial,
            "learned": learned,
            "learning_off_initial": off_initial,
            "learning_off_after": off_after,
        },
        "config": asdict(c),
        "interface_guards": {
            "recall_parameters": recall_parameters,
            "no_source_argument_at_inference": "source"
            not in inspect.signature(SourceMonitorCoresidencyGateV7.recall).parameters,
            "episode_activity_is_caller_supplied": True,
            "host_gain_normalization": False,
            "stronger_source_specific_drive": False,
            "host_confidence_scalar": False,
            "host_response_decision": False,
            "recall_settle_is_zero_input": True,
            "homeostasis_uses_local_firing_ema": True,
            "source_region_mask_is_hand_selected": True,
            "homeostasis_schedule_is_host_timed": True,
        },
        "seed_policy": {
            "non_scientific_smoke": SMOKE_SEED,
            "calibration": list(CALIBRATION_SEEDS),
            "development_reserved": list(DEVELOPMENT_SEEDS),
            "held_out": list(HELD_OUT_SEEDS),
        },
        "fixed_criteria": {
            "minimum_source_margin": MIN_SOURCE_MARGIN,
            "minimum_attribution_fraction": MIN_ATTRIBUTION_FRACTION,
            "unseen_source_spikes": 0,
            "learning_off_source_spikes": 0,
            "acceptance_rule": "bounded_loss_guard_floor_max_min",
            "loss_may_only_spend_surplus_above_floor": True,
            "weakest_source_margin_must_strictly_improve": True,
            "recall_settles_to_quiescence_before_read": True,
            "all_structural_and_lesion_components_required": True,
        },
        "honest_scope": (
            "V7 keeps v6's silent-by-construction settle-to-quiescence recall, v2's fixed local "
            "fast-spiking GABA-A competition, and the bounded-loss max-min acceptance rule with all "
            "twenty frozen components unchanged, and ADDS the shipped region-scoped intrinsic threshold "
            "homeostasis (Turrigiano activity-dependent excitability) on the source-memory populations, "
            "run during an explicit learning-off settling window so the least-active source's spike "
            "threshold is lowered and its recall is up-regulated. Sparse episode activity, physical "
            "source-afferent identity, the learning window, the pre-read settle timing, the hand-selected "
            "homeostasis regions, and the host-timed settling schedule remain developmental scaffolds. "
            "Spike counts and winners are host-read for evaluation only. The competition and homeostasis "
            "wiring is specified rather than self-organized; no language, confidence, or response policy is claimed."
        ),
        "elapsed_seconds": round(time.time() - t0, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one seed for source-monitor co-residency v7."
    )
    parser.add_argument("--seed", type=int, default=CALIBRATION_SEEDS[0])
    parser.add_argument("--phase", choices=tuple(PHASE_SEEDS), default="calibration")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    row = evaluate_calibration_seed(args.seed, phase=args.phase)
    print(
        "[source-monitor-coresidency-v7] "
        f"seed={row['seed']} phase={row['phase']} status={row['status']} "
        f"metrics={row['metrics']} components={row['components']}",
        flush=True,
    )
    if args.json:
        out_path = Path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(row, indent=2))
        print(f"[source-monitor-coresidency-v7] wrote {out_path}", flush=True)
    return 0 if row["status"].endswith("_PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
