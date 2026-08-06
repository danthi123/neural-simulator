"""Generalization v8 gate: v6 silent recall + Turrigiano SYNAPTIC SCALING.

v6 calibration was GO but development was a NO-GO on ONE of twenty frozen
components, ``weakest_source_margin_strictly_improved`` (seed 654): the fixed
symmetric GABA-A competition lifted the second-strongest source, not the
weakest, so ``min(M) > min(L)`` failed
(2026-08-06-source-monitor-coresidency-v6-development-NO-GO-...).

v7 tried the shipped INTRINSIC THRESHOLD homeostasis on the source pools and was
a WORSE NO-GO: the region-scoped homeostasis masks its pools so their spike is
read at the adapted SUB-THRESHOLD voltage, and switching the competing pools to
sub-threshold detection is incompatible with the v6 fixed GABA-A competition
(competition ON collapsed every margin to ~0.03 vs ~0.41 OFF)
(2026-08-06-source-monitor-coresidency-v7-development-NO-GO-...).  Its own
"next mechanism" names the surpass used here.

v8 = v6 UNCHANGED (silent-by-construction settle-to-quiescence recall, fixed
local fast-spiking GABA-A competition, and the identical bounded-loss max-min
acceptance rule with all twenty frozen components + thresholds) PLUS the shipped
Turrigiano synaptic scaling (``cfg.enable_synaptic_scaling``; Turrigiano 2008,
Cell 135:422; Turrigiano & Nelson 2004): after learning, an explicit
learning-off settling window multiplicatively up-regulates the UNDER-active
source's ``episode -> source`` recall synapses toward an activity set-point,
``w *= 1 + rate*(target - ema)`` per postsynaptic source-memory neuron.  Because
the mechanism touches ``cp_connections`` (real synapse weights) and NOT the
firing thresholds, the source-memory pools stay at PEAK Izhikevich-spike
detection, so the v6 competition keeps functioning exactly as calibrated -- the
precise failure mode v7 hit is avoided by construction.

Scoping is done with the shipped per-synapse plasticity-gain gate (Stage 1,
2026-04-27): during the settle window the gain vector is 1.0 on ONLY the
``episode -> source`` learned synapses and 0.0 elsewhere, so the scaling block
(``sim/bridge.py`` line ~10441, ``effective_scales = 1 + (scale-1)*gain``)
scales ONLY those synapses and leaves the afferent-identity, aPFC, ACC, and
competition weights byte-untouched.  No ``sim/`` edit is required.  Hebbian is
frozen during the window by setting its learning rate to 0 (the enable flag
stays True so the scaling weight-clip bounds remain ``[0, hebbian_max]`` and a
zero-weight control network scales 0 -> 0).

Seed 648 is reserved for non-scientific construction smoke.  Calibration seeds
650 and 651 are open for operating-point characterization.  Development opens
652/653/654; held-out opens 655/656/657 ONLY after development records an earned
GO here (``validate_phase_seed`` seals it otherwise).  Only the seed partition
advances; the circuit, thresholds, acceptance rule, and frozen scaling operating
point are held constant across development and held-out.  The runner does not
choose words, calculate confidence, or decide whether to respond.
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
from sim.backend import get_backend, to_host
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

# Aggregate generalization verdicts (written by aggregate_source_monitor_v8_seeds).
GENERALIZATION_DIR = Path("research/findings/raw/source_monitor_v8_generalization")
DEV_VERDICT_PATH = GENERALIZATION_DIR / "development_verdict.json"

# --- Frozen synaptic-scaling operating point -------------------------------
# The shipped synaptic scaling multiplies each postsynaptic neuron's incoming
# (gated) weights by 1 + rate*(target - ema) every step, clipped to +/-5%/step
# (sim/bridge.py ~10441; Turrigiano 2008).  We drive the learned episode->source
# routes under balanced, learning-off rehearsal so the WEAKEST source (lowest
# firing-rate EMA) accrues the largest up-scaling of its recall synapses.  These
# values are characterized and FROZEN on the calibration seeds (see the finding);
# development/held-out never touch them.
SCALING_TARGET_RATE = 0.02      # activity set-point (per-step firing fraction)
SCALING_EMA_ALPHA = 0.01        # tau_ema ~100 steps -> the rate estimate converges within the window
SCALING_RATE = 0.02             # per-step scaling gain (moves weak-source weights measurably, not saturating)
SCALING_SETTLING_STEPS = 5000   # balanced learning-off rehearsal


@dataclass(frozen=True)
class SourceMonitorConfigV8(SourceMonitorConfigV2):
    """Frozen v2/v6 operating point plus the frozen v8 synaptic-scaling operating point."""

    source_scaling_target_rate: float = SCALING_TARGET_RATE
    source_scaling_ema_alpha: float = SCALING_EMA_ALPHA
    source_scaling_rate: float = SCALING_RATE
    source_scaling_settling_steps: int = SCALING_SETTLING_STEPS


class SourceMonitorCoresidencyGateV8(SourceMonitorCoresidencyGateV6):
    """v6 silent-recall circuit with Turrigiano synaptic scaling on the recall synapses.

    NO homeostasis neuron mask is installed, so the source-memory pools keep
    Izhikevich PEAK spike detection and the v6 GABA-A competition is unchanged.
    Scaling runs ONLY inside an explicit ``settle_synaptic_scaling`` window, scoped
    to the ``episode -> source`` learned synapses via the per-synapse plasticity
    gain; during experience and recall no scaling runs, so recall reads a frozen
    network exactly as v6 does.
    """

    def __init__(
        self,
        *,
        seed: int,
        config: SourceMonitorConfigV8 | Mapping | None = None,
    ):
        c = (
            config
            if isinstance(config, SourceMonitorConfigV8)
            else SourceMonitorConfigV8(**(dict(config) if config else {}))
        )
        super().__init__(seed=seed, config=c)
        if self.bridge.cp_homeostasis_neuron_mask is not None:
            raise RuntimeError(
                "v8 must NOT mask the source pools (that is the v7 failure); "
                "the homeostasis neuron mask must stay None"
            )
        # Learned episode->source synapse positions (the recall synapses).
        self._learned_syn = np.asarray(
            self.bridge._plasticity_gate_to_synapses[SOURCE_LEARNING_GATE],
            dtype=np.int64,
        )

    def learned_weight_vector(self) -> np.ndarray:
        weights = np.asarray(to_host(self.bridge.cp_connections.data), dtype=np.float64)
        return weights[self._learned_syn].copy()

    def per_source_learned_l1(self) -> dict[str, float]:
        """L1 of each source's episode->source recall synapses.

        The learned gate lists the three sources' episode->source pathways in the
        canonical (pre,post)-sorted order; each source owns an equal contiguous
        third of the 6912 synapses (192 episode x 12 memory).  Split by that.
        """

        weights = self.learned_weight_vector()
        n = weights.size
        third = n // len(SOURCES)
        return {
            source: float(np.abs(weights[i * third : (i + 1) * third]).sum())
            for i, source in enumerate(SOURCES)
        }

    def source_threshold_vector(self) -> np.ndarray:
        thresholds = np.asarray(
            to_host(self.bridge.cp_neuron_firing_thresholds), dtype=np.float64
        )
        return np.concatenate(
            [thresholds[self._source_memory_indices[source]] for source in SOURCES]
        )

    def settle_synaptic_scaling(
        self,
        episode_patterns: Sequence[Sequence[int]],
        *,
        updates_enabled: bool,
    ) -> dict:
        """Balanced, learning-off rehearsal that scales the recall synapses.

        Hebbian is frozen (learning rate 0, enable flag left True so the scaling
        weight-clip bounds stay ``[hebbian_min=0, hebbian_max]``).  The scaling
        gain is 1.0 on ONLY the ``episode -> source`` learned synapses and 0.0
        elsewhere, so afferent-identity, aPFC, ACC, and competition weights are
        byte-untouched.  Competition is gated OFF during the window so the scale
        reflects each source's own learned recall drive, not rival inhibition.
        The least-active (weakest) source's recall synapses are scaled up the most
        -- the Turrigiano up-regulation of the under-active population.
        """

        xp, _ = get_backend()
        patterns = tuple(episode_patterns)
        if not patterns:
            raise ValueError("synaptic-scaling settling requires an episode pattern")
        c = self.config
        bridge = self.bridge
        cfg = bridge.core_config
        block_steps = int(c.training_steps) + int(c.rest_steps)
        cycles = int(
            math.ceil(
                int(c.source_scaling_settling_steps)
                / float(len(patterns) * block_steps)
            )
        )
        before_weights = self.learned_weight_vector()
        before_per_source = self.per_source_learned_l1()
        before_thresholds = self.source_threshold_vector()
        # Non-learned weights (everything except the recall synapses) must be
        # byte-untouched; snapshot them to prove it.
        all_before = np.asarray(to_host(bridge.cp_connections.data), dtype=np.float64)
        non_learned_mask = np.ones(all_before.size, dtype=bool)
        non_learned_mask[self._learned_syn] = False
        non_learned_before = all_before[non_learned_mask].copy()

        # Save mutated config + gate state.
        saved = {
            "enable_synaptic_scaling": cfg.enable_synaptic_scaling,
            "synaptic_scaling_rate": cfg.synaptic_scaling_rate,
            "homeostasis_target_rate": cfg.homeostasis_target_rate,
            "homeostasis_ema_alpha": cfg.homeostasis_ema_alpha,
            "hebbian_learning_rate": cfg.hebbian_learning_rate,
        }
        saved_gain = (
            None
            if bridge.cp_plasticity_rate_gain is None
            else bridge.cp_plasticity_rate_gain.copy()
        )
        saved_ema = (
            None
            if bridge.cp_neuron_activity_ema is None
            else bridge.cp_neuron_activity_ema.copy()
        )
        prior_comp = float(bridge._transmission_gate_values[SOURCE_COMPETITION_GATE])

        if updates_enabled:
            cfg.enable_synaptic_scaling = True
            cfg.synaptic_scaling_rate = float(c.source_scaling_rate)
            cfg.homeostasis_target_rate = float(c.source_scaling_target_rate)
            cfg.homeostasis_ema_alpha = float(c.source_scaling_ema_alpha)
            cfg.hebbian_learning_rate = 0.0  # freeze Hebbian; keep enable flag True for [0,max] clip
            # Scope scaling to ONLY the episode->source recall synapses.
            gain = xp.zeros_like(bridge.cp_plasticity_rate_gain)
            gain[xp.asarray(self._learned_syn)] = xp.float32(1.0)
            bridge.cp_plasticity_rate_gain = gain
            # Start the rate estimate at the set-point so there is no initial transient.
            bridge.cp_neuron_activity_ema[:] = xp.float32(c.source_scaling_target_rate)

        # Competition OFF during the window so the scale reflects each source's
        # own learned recall drive, not rival inhibition.
        bridge.set_transmission_gate(SOURCE_COMPETITION_GATE, 0.0)
        try:
            for _ in range(cycles):
                for pattern in patterns:
                    episode_global = self._episode_global_indices(pattern)
                    self._drive(episode_global)
                    for _ in range(int(c.training_steps)):
                        bridge._run_one_simulation_step()
                    # Rest at zero input (competition already off).
                    bridge.cp_external_input_current[:] = 0.0
                    for _ in range(int(c.rest_steps)):
                        bridge._run_one_simulation_step()
        finally:
            # Restore config + gate + ema, then competition.
            cfg.enable_synaptic_scaling = saved["enable_synaptic_scaling"]
            cfg.synaptic_scaling_rate = saved["synaptic_scaling_rate"]
            cfg.homeostasis_target_rate = saved["homeostasis_target_rate"]
            cfg.homeostasis_ema_alpha = saved["homeostasis_ema_alpha"]
            cfg.hebbian_learning_rate = saved["hebbian_learning_rate"]
            if saved_gain is not None:
                bridge.cp_plasticity_rate_gain = saved_gain
            if saved_ema is not None:
                bridge.cp_neuron_activity_ema = saved_ema
            bridge.set_transmission_gate(SOURCE_COMPETITION_GATE, prior_comp)
            bridge.cp_external_input_current[:] = 0.0

        after_weights = self.learned_weight_vector()
        after_per_source = self.per_source_learned_l1()
        after_thresholds = self.source_threshold_vector()
        all_after = np.asarray(to_host(bridge.cp_connections.data), dtype=np.float64)
        non_learned_after = all_after[non_learned_mask]

        weight_delta = after_weights - before_weights
        # The weakest (lowest-L1) source BEFORE settling; did its recall weights grow?
        weakest_source = min(before_per_source, key=before_per_source.get)
        weakest_l1_gain = after_per_source[weakest_source] - before_per_source[weakest_source]
        per_source_l1_gain = {
            source: after_per_source[source] - before_per_source[source]
            for source in SOURCES
        }
        # Multiplicative scaling toward a common set-point EQUALIZES firing, so the
        # honest direction check is the weakest source's SHARE of the total recall
        # weight (relative up-regulation), not an absolute L1 increase.
        tot_before = sum(before_per_source.values()) or 1.0
        tot_after = sum(after_per_source.values()) or 1.0
        before_share = {s: before_per_source[s] / tot_before for s in SOURCES}
        after_share = {s: after_per_source[s] / tot_after for s in SOURCES}
        weakest_share_gain = after_share[weakest_source] - before_share[weakest_source]
        return {
            "updates_enabled": bool(updates_enabled),
            "patterns_per_cycle": len(patterns),
            "cycles": cycles,
            "executed_steps": cycles * len(patterns) * block_steps,
            "weight_l1_before": float(np.abs(before_weights).sum()),
            "weight_l1_after": float(np.abs(after_weights).sum()),
            "weight_delta_l1": float(np.abs(weight_delta).sum()),
            "weight_delta_max_abs": float(np.abs(weight_delta).max()),
            "before_per_source_l1": before_per_source,
            "after_per_source_l1": after_per_source,
            "per_source_l1_gain": per_source_l1_gain,
            "weakest_source_before": weakest_source,
            "weakest_source_l1_gain": float(weakest_l1_gain),
            "before_share": before_share,
            "after_share": after_share,
            "weakest_source_share_gain": float(weakest_share_gain),
            "weight_changed": bool(float(np.abs(weight_delta).sum()) > 0.0),
            "non_source_weights_unchanged": bool(
                np.array_equal(non_learned_before, non_learned_after)
            ),
            "source_thresholds_unchanged": bool(
                np.array_equal(before_thresholds, after_thresholds)
            ),
            "homeostasis_mask_stays_none": bool(
                self.bridge.cp_homeostasis_neuron_mask is None
            ),
        }


def _development_is_go() -> bool:
    """True only if the v8 development aggregate verdict exists and reads GO."""

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
            f"phase {phase!r} is not a v8 phase; choose from {tuple(PHASE_SEEDS)}"
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
            f"seed {seed} is not a v8 {phase} seed; allowed={allowed}, "
            f"smoke={SMOKE_SEED}, calibration={CALIBRATION_SEEDS}, "
            f"development={DEVELOPMENT_SEEDS}, held_out={HELD_OUT_SEEDS}"
        )
    return seed


def evaluate_calibration_seed(
    seed: int,
    config: SourceMonitorConfigV8 | None = None,
    *,
    phase: str = "calibration",
) -> dict:
    """Run one seed of ``phase`` and all preregistered controls.

    Identical to v6's evaluator except that, after learning, the intact (and
    swapped) network runs a balanced learning-off settling window in which
    Turrigiano synaptic scaling multiplicatively up-regulates the weakest
    source's episode->source recall synapses.  The twenty frozen v6 components,
    thresholds, and acceptance rule are unchanged; the scaling-integrity checks
    are recorded as preconditions, not as new components.
    """

    seed = validate_phase_seed(seed, phase)
    c = config or SourceMonitorConfigV8()
    patterns = make_episode_patterns(seed, 5, c)
    t0 = time.time()

    intact = SourceMonitorCoresidencyGateV8(seed=seed, config=c)
    initial = intact.weight_summary()
    intact.experience(patterns[0], visual_activity=True)
    intact.experience(patterns[1], auditory_activity=True)
    intact.experience(patterns[2], corollary_discharge=True)
    intact.experience(patterns[3], visual_activity=True, auditory_activity=True)
    learned = intact.weight_summary()
    # v8 addition: Turrigiano synaptic scaling up-regulates the weakest source.
    scaling_settling = intact.settle_synaptic_scaling(patterns[:3], updates_enabled=True)
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

    swapped = SourceMonitorCoresidencyGateV8(seed=seed + 10000, config=c)
    swapped.experience(patterns[0], auditory_activity=True)
    swapped.experience(patterns[1], visual_activity=True)
    swapped.settle_synaptic_scaling(patterns[:2], updates_enabled=True)
    swap_zero = swapped.recall(patterns[0])
    swap_one = swapped.recall(patterns[1])

    learning_off = SourceMonitorCoresidencyGateV8(seed=seed + 20000, config=c)
    off_initial = learning_off.weight_summary()
    learning_off.experience(patterns[0], visual_activity=True, learning_enabled=False)
    off_after = learning_off.weight_summary()
    learning_off.settle_synaptic_scaling(patterns[:1], updates_enabled=True)
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
    recall_parameters = list(inspect.signature(SourceMonitorCoresidencyGateV8.recall).parameters)
    earned = Verdict(f"source-monitor co-residency v8 {phase}")
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
        "v8 preserves v1 source input strength",
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
        "source pools stay at PEAK spike detection (no homeostasis mask, thresholds unchanged)",
        scaling_settling["homeostasis_mask_stays_none"]
        and scaling_settling["source_thresholds_unchanged"],
        expect=True,
    )
    earned.require(
        "synaptic scaling is active and scoped: recall-synapse weights changed, non-source weights byte-untouched",
        scaling_settling["weight_changed"]
        and scaling_settling["non_source_weights_unchanged"],
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
        "source synaptic-scaling rate",
        requested=c.source_scaling_rate,
        applied=c.source_scaling_rate,
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
        "weakest-source recall synaptic scaling",
        before=0.0,
        after=float(scaling_settling["weakest_source_l1_gain"]),
    )
    earned.disabled(
        "STDP, reward modulation, intrinsic threshold homeostasis, short-term plasticity, and structural plasticity",
        why=(
            "v8 isolates Hebbian source association, fixed local GABA-A competition, "
            "and Turrigiano synaptic scaling of the source recall synapses"
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
        "scaling": {
            "intact_settling": scaling_settling,
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
            not in inspect.signature(SourceMonitorCoresidencyGateV8.recall).parameters,
            "episode_activity_is_caller_supplied": True,
            "host_gain_normalization": False,
            "stronger_source_specific_drive": False,
            "host_confidence_scalar": False,
            "host_response_decision": False,
            "recall_settle_is_zero_input": True,
            "scaling_acts_on_real_synapse_weights": True,
            "scaling_scoped_to_recall_synapses": True,
            "scaling_schedule_is_host_timed": True,
            "source_pools_stay_at_peak_detection": True,
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
            "V8 keeps v6's silent-by-construction settle-to-quiescence recall, v2's fixed local "
            "fast-spiking GABA-A competition, and the bounded-loss max-min acceptance rule with all "
            "twenty frozen components unchanged, and ADDS the shipped Turrigiano synaptic scaling "
            "(cfg.enable_synaptic_scaling; Turrigiano 2008) run during an explicit learning-off settling "
            "window to multiplicatively up-regulate the weakest source's episode->source recall synapses. "
            "The scaling acts on real synapse weights (cp_connections) and NOT on firing thresholds, so the "
            "source pools stay at PEAK Izhikevich-spike detection and the v6 competition is unchanged -- the "
            "v7 sub-threshold-masking failure is avoided by construction. Sparse episode activity, physical "
            "source-afferent identity, the learning window, the pre-read settle timing, the scoping of scaling "
            "to the recall synapses, and the host-timed settling schedule remain developmental scaffolds. "
            "Spike counts and winners are host-read for evaluation only. The competition and scaling wiring is "
            "specified rather than self-organized; no language, confidence, or response policy is claimed."
        ),
        "elapsed_seconds": round(time.time() - t0, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one seed for source-monitor co-residency v8."
    )
    parser.add_argument("--seed", type=int, default=CALIBRATION_SEEDS[0])
    parser.add_argument("--phase", choices=tuple(PHASE_SEEDS), default="calibration")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    row = evaluate_calibration_seed(args.seed, phase=args.phase)
    print(
        "[source-monitor-coresidency-v8] "
        f"seed={row['seed']} phase={row['phase']} status={row['status']} "
        f"metrics={row['metrics']} components={row['components']}",
        flush=True,
    )
    if args.json:
        out_path = Path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(row, indent=2))
        print(f"[source-monitor-coresidency-v8] wrote {out_path}", flush=True)
    return 0 if row["status"].endswith("_PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
