"""Generalization v9 gate: v6 silent recall + Vogels-Sprekeler INHIBITORY STDP.

v6 calibration was GO but development a NO-GO on ONE of twenty frozen
components, ``weakest_source_margin_strictly_improved``: the fixed symmetric
GABA-A competition lifted the second-strongest source, not the weakest, so
``min(M) > min(L)`` failed
(2026-08-06-source-monitor-coresidency-v6-development-NO-GO-...).

v7 (intrinsic THRESHOLD homeostasis) and v8 (Turrigiano SYNAPTIC SCALING) both
NO-GO: each defends an ACTIVITY LEVEL, but the criterion measures a per-source
MARGIN = the CONTRAST between the correct source and its rivals.  v8's finding
names the surpass used here: target the CONTRAST via INHIBITORY plasticity, not
the firing rate -- up-regulate the under-margin source's RIVAL inhibition, not
the excitatory recall drive
(2026-08-06-source-monitor-coresidency-v8-development-NO-GO-...).

v9 = v6 UNCHANGED (silent-by-construction settle-to-quiescence recall, fixed
local fast-spiking GABA-A competition, and the identical bounded-loss max-min
acceptance rule with all twenty frozen components + thresholds) PLUS the shipped
Vogels-Sprekeler inhibitory STDP (``cfg.enable_inhibitory_stdp``; Vogels et al.
2011, Science 334:1569) run in an explicit learning-off rehearsal window on ONLY
the ``interneuron -> rival`` GABA-A synapses.  The rule strengthens rival
inhibition toward a target E/I set-point where a rival is over-active during a
source's recall, so the under-margin source suppresses its rivals MORE, raising
its margin -- a BETWEEN-SOURCE contrast -- WITHOUT touching the excitatory source
pools (they stay at PEAK Izhikevich detection, the code uncompressed).

Scoping: during the rehearsal window ``enable_hebbian_learning`` is set False and
``enable_inhibitory_stdp`` True (the shipped composability validator forbids the
combination; v6's excitatory Hebbian is off during the window and restored
after).  The per-synapse plasticity-gain vector is 1.0 on ONLY the
``interneuron -> rival`` GABA-A synapses and 0.0 elsewhere, and those synapses are
temporarily added to the per-synapse plastic mask, so the Vogels update
(``sim/bridge.py`` ``_apply_inhibitory_stdp``) touches ONLY the competition
inhibition and leaves the episode->source recall, afferent-identity, aPFC, and
ACC weights byte-untouched.  No ``sim/`` edit is required; the v6 build is
byte-identical (the interneuron->rival routes are built plastic=False and the
plastic mask / gain are set only inside the window and restored on exit).

Seed 648 is reserved for non-scientific construction smoke.  Calibration seeds
650 and 651 are open for operating-point characterization.  Development opens
652/653/654; held-out opens 655/656/657 ONLY after development records an earned
GO here (``validate_phase_seed`` seals it otherwise).  Only the seed partition
advances; the circuit, thresholds, acceptance rule, and frozen ISP operating
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
from dataclasses import asdict, dataclass
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

# Aggregate generalization verdicts (written by aggregate_source_monitor_v9_seeds).
GENERALIZATION_DIR = Path("research/findings/raw/source_monitor_v9_generalization")
DEV_VERDICT_PATH = GENERALIZATION_DIR / "development_verdict.json"

# --- Frozen Vogels-Sprekeler inhibitory-STDP operating point ---------------
# The shipped rule (sim/bridge.py _apply_inhibitory_stdp; Vogels 2011) updates
# each plastic inhibitory GABA-A synapse by dw = eta*(post_trace - alpha) on an
# inhibitory presynaptic spike and dw = eta*pre_trace on a postsynaptic spike,
# with alpha = 2*target_rate*tau_steps the E/I set-point.  We rehearse the
# learned recall (episode drive, no afferents) with competition ON so the rule
# tunes the interneuron->rival inhibition against the recall-time rival leak.
# These values are characterized and FROZEN on the calibration seeds (see the
# finding); development/held-out never touch them.
ISTDP_TAU_MS = 20.0             # Vogels 2011 inhibitory trace time constant
ISTDP_TARGET_RATE = 0.02        # E/I set-point (per-step firing fraction)
ISTDP_ETA = 0.02                # per-event inhibitory learning rate
ISTDP_W_MIN = 0.0
ISTDP_W_MAX = 6.0               # >= the fixed interneuron_to_rival_weight (3.0)
ISTDP_SETTLING_STEPS = 5000     # balanced learning-off rehearsal of the learned recall


@dataclass(frozen=True)
class SourceMonitorConfigV9(SourceMonitorConfigV2):
    """Frozen v2/v6 operating point plus the frozen v9 inhibitory-STDP operating point."""

    istdp_tau_ms: float = ISTDP_TAU_MS
    istdp_target_rate: float = ISTDP_TARGET_RATE
    istdp_eta: float = ISTDP_ETA
    istdp_w_min: float = ISTDP_W_MIN
    istdp_w_max: float = ISTDP_W_MAX
    istdp_settling_steps: int = ISTDP_SETTLING_STEPS


class SourceMonitorCoresidencyGateV9(SourceMonitorCoresidencyGateV6):
    """v6 silent-recall circuit with Vogels-Sprekeler iSTDP on the rival inhibition.

    NO homeostasis neuron mask is installed, so the source-memory pools keep
    Izhikevich PEAK spike detection and the v6 GABA-A competition is unchanged.
    Inhibitory STDP runs ONLY inside an explicit ``settle_inhibitory_stdp``
    window, scoped to the ``interneuron -> rival`` GABA-A synapses via the
    per-synapse plastic mask + plasticity gain; during experience and recall no
    iSTDP runs, so recall reads a frozen network exactly as v6 does.
    """

    def __init__(
        self,
        *,
        seed: int,
        config: SourceMonitorConfigV9 | Mapping | None = None,
    ):
        c = (
            config
            if isinstance(config, SourceMonitorConfigV9)
            else SourceMonitorConfigV9(**(dict(config) if config else {}))
        )
        super().__init__(seed=seed, config=c)
        if self.bridge.cp_homeostasis_neuron_mask is not None:
            raise RuntimeError(
                "v9 must NOT mask the source pools (that is the v7 failure); "
                "the homeostasis neuron mask must stay None"
            )
        # Locate the interneuron->rival GABA-A synapses in cp_connections/COO order.
        # Every FS interneuron projects ONLY to rival source-memory pools, so a
        # synapse whose presynaptic neuron is a competition interneuron and whose
        # postsynaptic neuron is a source-memory neuron is exactly one of these.
        coo = self.bridge._get_cached_coo()
        rows = np.asarray(to_host(coo.row), dtype=np.int64)
        cols = np.asarray(to_host(coo.col), dtype=np.int64)
        inter_all = np.concatenate(
            [self._competition_indices[s] for s in SOURCES]
        )
        mem_all = np.concatenate(
            [self._source_memory_indices[s] for s in SOURCES]
        )
        rival_mask = np.isin(rows, inter_all) & np.isin(cols, mem_all)
        self._rival_syn = np.where(rival_mask)[0].astype(np.int64)
        # Per-source ownership of the rival-inhibition synapses (by presynaptic
        # interneuron): source S owns synapse i iff rows[i] in S's interneurons.
        self._rival_syn_source = {
            s: np.where(np.isin(rows[self._rival_syn], self._competition_indices[s]))[0]
            for s in SOURCES
        }

    def rival_weight_vector(self) -> np.ndarray:
        weights = np.asarray(to_host(self.bridge.cp_connections.data), dtype=np.float64)
        return weights[self._rival_syn].copy()

    def per_source_rival_l1(self) -> dict[str, float]:
        """L1 of each source's interneuron->rival GABA-A synapses."""

        weights = self.rival_weight_vector()
        return {
            s: float(np.abs(weights[self._rival_syn_source[s]]).sum())
            for s in SOURCES
        }

    def source_threshold_vector(self) -> np.ndarray:
        thresholds = np.asarray(
            to_host(self.bridge.cp_neuron_firing_thresholds), dtype=np.float64
        )
        return np.concatenate(
            [thresholds[self._source_memory_indices[source]] for source in SOURCES]
        )

    def settle_inhibitory_stdp(
        self,
        episode_patterns: Sequence[Sequence[int]],
        *,
        updates_enabled: bool,
    ) -> dict:
        """Balanced, learning-off recall rehearsal that tunes the rival inhibition.

        v6 Hebbian is frozen (enable flag set False for the window, restored on
        exit) so the shipped composability validator accepts inhibitory STDP.
        The plasticity gain is 1.0 on ONLY the ``interneuron -> rival`` GABA-A
        synapses and 0.0 elsewhere, and those synapses are temporarily added to
        the per-synapse plastic mask, so the Vogels update touches ONLY the rival
        inhibition; episode->source recall, afferent, aPFC, and ACC weights are
        byte-untouched.  Competition is ON during the window so the interneurons
        fire and the rule sees the recall-time rival leak it is meant to suppress.
        """

        xp, _ = get_backend()
        patterns = tuple(episode_patterns)
        if not patterns:
            raise ValueError("inhibitory-STDP settling requires an episode pattern")
        c = self.config
        bridge = self.bridge
        cfg = bridge.core_config
        block_steps = int(c.training_steps) + int(c.rest_steps)
        cycles = int(
            math.ceil(
                int(c.istdp_settling_steps)
                / float(len(patterns) * block_steps)
            )
        )

        before_weights = self.rival_weight_vector()
        before_per_source = self.per_source_rival_l1()
        before_thresholds = self.source_threshold_vector()
        # Everything except the rival-inhibition synapses must be byte-untouched.
        all_before = np.asarray(to_host(bridge.cp_connections.data), dtype=np.float64)
        non_rival_mask = np.ones(all_before.size, dtype=bool)
        non_rival_mask[self._rival_syn] = False
        non_rival_before = all_before[non_rival_mask].copy()

        saved = {
            "enable_inhibitory_stdp": cfg.enable_inhibitory_stdp,
            "enable_hebbian_learning": cfg.enable_hebbian_learning,
            "inhibitory_stdp_tau_ms": cfg.inhibitory_stdp_tau_ms,
            "inhibitory_stdp_target_rate_per_step": cfg.inhibitory_stdp_target_rate_per_step,
            "inhibitory_stdp_eta": cfg.inhibitory_stdp_eta,
            "inhibitory_stdp_w_min": cfg.inhibitory_stdp_w_min,
            "inhibitory_stdp_w_max": cfg.inhibitory_stdp_w_max,
        }
        saved_gain = (
            None
            if bridge.cp_plasticity_rate_gain is None
            else bridge.cp_plasticity_rate_gain.copy()
        )
        saved_plastic = (
            None
            if bridge.cp_synapse_plastic_mask is None
            else bridge.cp_synapse_plastic_mask.copy()
        )
        saved_trace = (
            None
            if bridge.cp_inhibitory_stdp_trace is None
            else bridge.cp_inhibitory_stdp_trace.copy()
        )
        prior_comp = float(bridge._transmission_gate_values[SOURCE_COMPETITION_GATE])

        if updates_enabled:
            cfg.enable_hebbian_learning = False
            cfg.enable_inhibitory_stdp = True
            cfg.inhibitory_stdp_tau_ms = float(c.istdp_tau_ms)
            cfg.inhibitory_stdp_target_rate_per_step = float(c.istdp_target_rate)
            cfg.inhibitory_stdp_eta = float(c.istdp_eta)
            cfg.inhibitory_stdp_w_min = float(c.istdp_w_min)
            cfg.inhibitory_stdp_w_max = float(c.istdp_w_max)
            # Allocate the per-neuron inhibitory trace (the shipped step reads it).
            n_neurons = int(bridge.cp_firing_states.size)
            bridge.cp_inhibitory_stdp_trace = xp.zeros(n_neurons, dtype=xp.float32)
            # Scope the update to ONLY the interneuron->rival synapses.
            gain = xp.zeros_like(bridge.cp_plasticity_rate_gain)
            gain[xp.asarray(self._rival_syn)] = xp.float32(1.0)
            bridge.cp_plasticity_rate_gain = gain
            plastic = bridge.cp_synapse_plastic_mask.copy()
            plastic[xp.asarray(self._rival_syn)] = True
            bridge.cp_synapse_plastic_mask = plastic

        # Competition ON so interneurons fire and the rule sees the rival leak.
        bridge.set_transmission_gate(SOURCE_COMPETITION_GATE, 1.0)
        try:
            for _ in range(cycles):
                for pattern in patterns:
                    episode_global = self._episode_global_indices(pattern)
                    # Recall-like drive: episode only, NO source afferents, so the
                    # rule rehearses exactly the recall-time competition dynamics.
                    self._drive(episode_global)
                    for _ in range(int(c.training_steps)):
                        bridge._run_one_simulation_step()
                    bridge.cp_external_input_current[:] = 0.0
                    for _ in range(int(c.rest_steps)):
                        bridge._run_one_simulation_step()
        finally:
            cfg.enable_inhibitory_stdp = saved["enable_inhibitory_stdp"]
            cfg.enable_hebbian_learning = saved["enable_hebbian_learning"]
            cfg.inhibitory_stdp_tau_ms = saved["inhibitory_stdp_tau_ms"]
            cfg.inhibitory_stdp_target_rate_per_step = saved[
                "inhibitory_stdp_target_rate_per_step"
            ]
            cfg.inhibitory_stdp_eta = saved["inhibitory_stdp_eta"]
            cfg.inhibitory_stdp_w_min = saved["inhibitory_stdp_w_min"]
            cfg.inhibitory_stdp_w_max = saved["inhibitory_stdp_w_max"]
            if saved_gain is not None:
                bridge.cp_plasticity_rate_gain = saved_gain
            if saved_plastic is not None:
                bridge.cp_synapse_plastic_mask = saved_plastic
            bridge.cp_inhibitory_stdp_trace = saved_trace
            bridge.set_transmission_gate(SOURCE_COMPETITION_GATE, prior_comp)
            bridge.cp_external_input_current[:] = 0.0

        after_weights = self.rival_weight_vector()
        after_per_source = self.per_source_rival_l1()
        after_thresholds = self.source_threshold_vector()
        all_after = np.asarray(to_host(bridge.cp_connections.data), dtype=np.float64)
        non_rival_after = all_after[non_rival_mask]

        weight_delta = after_weights - before_weights
        per_source_l1_gain = {
            s: after_per_source[s] - before_per_source[s] for s in SOURCES
        }
        return {
            "updates_enabled": bool(updates_enabled),
            "patterns_per_cycle": len(patterns),
            "cycles": cycles,
            "executed_steps": cycles * len(patterns) * block_steps,
            "rival_syn_count": int(self._rival_syn.size),
            "weight_l1_before": float(np.abs(before_weights).sum()),
            "weight_l1_after": float(np.abs(after_weights).sum()),
            "weight_delta_l1": float(np.abs(weight_delta).sum()),
            "weight_delta_max_abs": float(np.abs(weight_delta).max()),
            "before_per_source_l1": before_per_source,
            "after_per_source_l1": after_per_source,
            "per_source_l1_gain": per_source_l1_gain,
            "weight_changed": bool(float(np.abs(weight_delta).sum()) > 0.0),
            "non_rival_weights_unchanged": bool(
                np.array_equal(non_rival_before, non_rival_after)
            ),
            "source_thresholds_unchanged": bool(
                np.array_equal(before_thresholds, after_thresholds)
            ),
            "homeostasis_mask_stays_none": bool(
                self.bridge.cp_homeostasis_neuron_mask is None
            ),
        }


def _development_is_go() -> bool:
    """True only if the v9 development aggregate verdict exists and reads GO."""

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
            f"phase {phase!r} is not a v9 phase; choose from {tuple(PHASE_SEEDS)}"
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
            f"seed {seed} is not a v9 {phase} seed; allowed={allowed}, "
            f"smoke={SMOKE_SEED}, calibration={CALIBRATION_SEEDS}, "
            f"development={DEVELOPMENT_SEEDS}, held_out={HELD_OUT_SEEDS}"
        )
    return seed


def evaluate_calibration_seed(
    seed: int,
    config: SourceMonitorConfigV9 | None = None,
    *,
    phase: str = "calibration",
) -> dict:
    """Run one seed of ``phase`` and all preregistered controls.

    Identical to v6's evaluator except that, after learning, the intact (and
    swapped) network runs a balanced learning-off recall rehearsal in which
    Vogels-Sprekeler inhibitory STDP tunes the interneuron->rival GABA-A
    synapses toward the E/I set-point.  The twenty frozen v6 components,
    thresholds, and acceptance rule are unchanged; the iSTDP-integrity checks are
    recorded as preconditions, not as new components.
    """

    seed = validate_phase_seed(seed, phase)
    c = config or SourceMonitorConfigV9()
    patterns = make_episode_patterns(seed, 5, c)
    t0 = time.time()

    intact = SourceMonitorCoresidencyGateV9(seed=seed, config=c)
    initial = intact.weight_summary()
    intact.experience(patterns[0], visual_activity=True)
    intact.experience(patterns[1], auditory_activity=True)
    intact.experience(patterns[2], corollary_discharge=True)
    intact.experience(patterns[3], visual_activity=True, auditory_activity=True)
    learned = intact.weight_summary()
    # v9 addition: Vogels-Sprekeler iSTDP tunes the rival inhibition.
    istdp_settling = intact.settle_inhibitory_stdp(patterns[:3], updates_enabled=True)
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

    swapped = SourceMonitorCoresidencyGateV9(seed=seed + 10000, config=c)
    swapped.experience(patterns[0], auditory_activity=True)
    swapped.experience(patterns[1], visual_activity=True)
    swapped.settle_inhibitory_stdp(patterns[:2], updates_enabled=True)
    swap_zero = swapped.recall(patterns[0])
    swap_one = swapped.recall(patterns[1])

    learning_off = SourceMonitorCoresidencyGateV9(seed=seed + 20000, config=c)
    off_initial = learning_off.weight_summary()
    learning_off.experience(patterns[0], visual_activity=True, learning_enabled=False)
    off_after = learning_off.weight_summary()
    learning_off.settle_inhibitory_stdp(patterns[:1], updates_enabled=True)
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
    # The recall-time rival burden the inhibition is meant to reduce: sum of the
    # non-dominant source rates during each source's own recall (0 => nothing for
    # inhibitory plasticity to act on; the v4 wall).
    rival_burden = {
        "seen": sum(seen["source_rates"][s] for s in SOURCES if s != "seen"),
        "heard": sum(heard["source_rates"][s] for s in SOURCES if s != "heard"),
        "self_generated": sum(
            self_generated["source_rates"][s] for s in SOURCES if s != "self_generated"
        ),
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
    recall_parameters = list(inspect.signature(SourceMonitorCoresidencyGateV9.recall).parameters)
    earned = Verdict(f"source-monitor co-residency v9 {phase}")
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
        "v9 preserves v1 source input strength",
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
        istdp_settling["homeostasis_mask_stays_none"]
        and istdp_settling["source_thresholds_unchanged"],
        expect=True,
    )
    earned.require(
        "inhibitory STDP is active and scoped: interneuron->rival weights changed, non-competition weights byte-untouched",
        istdp_settling["weight_changed"]
        and istdp_settling["non_rival_weights_unchanged"],
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
        "inhibitory-STDP learning rate",
        requested=c.istdp_eta,
        applied=c.istdp_eta,
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
        "rival-inhibition inhibitory STDP",
        before=float(istdp_settling["weight_l1_before"]),
        after=float(istdp_settling["weight_l1_after"]),
    )
    earned.disabled(
        "ordinary STDP, reward modulation, intrinsic threshold homeostasis, synaptic scaling, short-term plasticity, and structural plasticity",
        why=(
            "v9 isolates Hebbian source association, fixed local GABA-A competition, "
            "and Vogels-Sprekeler inhibitory STDP of the interneuron->rival synapses"
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
            "recall_rival_burden": rival_burden,
            "bounded_loss": bounded_loss,
            "spendable_surplus": spendable_surplus,
            "minimum_lesion_margin": min(lesion_margins.values()),
            "learning_off_source_spikes": float(sum(off_recall["source_spikes"].values())),
            "learning_off_settle_steps": int(off_recall["settle"]["settle_steps"]),
        },
        "istdp": {
            "intact_settling": istdp_settling,
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
            not in inspect.signature(SourceMonitorCoresidencyGateV9.recall).parameters,
            "episode_activity_is_caller_supplied": True,
            "host_gain_normalization": False,
            "stronger_source_specific_drive": False,
            "host_confidence_scalar": False,
            "host_response_decision": False,
            "recall_settle_is_zero_input": True,
            "istdp_acts_on_real_synapse_weights": True,
            "istdp_scoped_to_rival_inhibition": True,
            "istdp_schedule_is_host_timed": True,
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
            "V9 keeps v6's silent-by-construction settle-to-quiescence recall, v2's fixed local "
            "fast-spiking GABA-A competition, and the bounded-loss max-min acceptance rule with all "
            "twenty frozen components unchanged, and ADDS the shipped Vogels-Sprekeler inhibitory STDP "
            "(cfg.enable_inhibitory_stdp; Vogels 2011) run during an explicit learning-off recall "
            "rehearsal to tune the interneuron->rival GABA-A synapses toward an E/I set-point. The rule "
            "acts on real GABA-A synapse weights (cp_connections) and NOT on firing thresholds or the "
            "excitatory recall synapses, so the source pools stay at PEAK Izhikevich detection and the "
            "excitatory code is uncompressed -- the v7 masking and v8 compression failures are avoided by "
            "construction. Sparse episode activity, physical source-afferent identity, the learning window, "
            "the pre-read settle timing, the scoping of iSTDP to the rival-inhibition synapses, and the "
            "host-timed rehearsal schedule remain developmental scaffolds. Spike counts and winners are "
            "host-read for evaluation only. The competition and iSTDP wiring is specified rather than "
            "self-organized; no language, confidence, or response policy is claimed."
        ),
        "elapsed_seconds": round(time.time() - t0, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one seed for source-monitor co-residency v9."
    )
    parser.add_argument("--seed", type=int, default=CALIBRATION_SEEDS[0])
    parser.add_argument("--phase", choices=tuple(PHASE_SEEDS), default="calibration")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    row = evaluate_calibration_seed(args.seed, phase=args.phase)
    print(
        "[source-monitor-coresidency-v9] "
        f"seed={row['seed']} phase={row['phase']} status={row['status']} "
        f"metrics={row['metrics']} components={row['components']}",
        flush=True,
    )
    if args.json:
        out_path = Path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(row, indent=2))
        print(f"[source-monitor-coresidency-v9] wrote {out_path}", flush=True)
    return 0 if row["status"].endswith("_PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
