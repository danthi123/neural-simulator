"""Calibration-only v2 gate for co-resident episodic source monitoring.

V1 failed development because one fresh substrate produced a heard-source
margin of 0.11 against the fixed 0.15 floor.  This attempt leaves that floor,
the physical source afferents, and the zero-initialized learned routes intact.
It adds a local biased-competition motif: each source-memory population
recruits fast-spiking interneurons that inhibit the other two source pools.
Competition is a neural circuit on the same bridge, not host normalization or
stronger source-specific input.

Only calibration seeds 216 and 217 are open.  Development seeds 218, 219, and
314 and held-out seeds 315, 316, and 317 are named but mechanically rejected.
The runner does not choose words, calculate confidence, or decide whether to
respond.
"""
from __future__ import annotations

import argparse
import inspect
import json
import logging
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
    DEVELOPMENT_MIN_ATTRIBUTION_FRACTION,
    DEVELOPMENT_MIN_SOURCE_MARGIN,
    EPISODE_REGION,
    SOURCE_AFFERENT,
    SOURCE_AFFERENT_GATE,
    SOURCE_LEARNING_GATE,
    SOURCE_MEMORY,
    SOURCE_RECALL_GATE,
    SOURCES,
    SourceMonitorConfig,
    SourceMonitorCoresidencyGate,
    _dominant_source,
    _source_margin,
    make_episode_patterns,
)
from sim.backend import to_host
from sim.bridge import SimulationBridge
from sim.config import CoreSimConfig, GPUConfig, RuntimeState, VisualizationConfig
from sim.regions import BrainRegion, RegionPathway
from tools.lab import attributable_to
from tools.verdict import UNDEFINED, Verdict


CALIBRATION_SEEDS = (216, 217)
DEVELOPMENT_SEEDS = (218, 219, 314)
HELD_OUT_SEEDS = (315, 316, 317)

MIN_SOURCE_MARGIN = DEVELOPMENT_MIN_SOURCE_MARGIN
MIN_ATTRIBUTION_FRACTION = DEVELOPMENT_MIN_ATTRIBUTION_FRACTION

SOURCE_INTERNEURON = {source: f"source_competition_fs_{source}" for source in SOURCES}
SOURCE_COMPETITION_GATE = "source_pool_competition"


@dataclass(frozen=True)
class SourceMonitorConfigV2(SourceMonitorConfig):
    """V1 operating point plus a small source-local inhibitory circuit."""

    n_source_interneuron: int = 6
    source_to_interneuron_weight: float = 1.0
    interneuron_to_rival_weight: float = 3.0


class SourceMonitorCoresidencyGateV2(SourceMonitorCoresidencyGate):
    """V1 source memory with local cross-pool inhibitory stabilization."""

    def __init__(
        self,
        *,
        seed: int,
        config: SourceMonitorConfigV2 | Mapping | None = None,
    ):
        c = (
            config
            if isinstance(config, SourceMonitorConfigV2)
            else SourceMonitorConfigV2(**(dict(config) if config else {}))
        )
        super().__init__(seed=seed, config=c)
        rm = self.bridge.region_manager
        self._competition_indices = {
            source: np.asarray(rm.indices(SOURCE_INTERNEURON[source]), dtype=np.int64)
            for source in SOURCES
        }
        self.bridge.set_transmission_gate(SOURCE_COMPETITION_GATE, 1.0)

    def _rest(self) -> None:
        """Drain trial state without allowing an inhibitory rebound loop."""

        competition_gain = float(
            self.bridge._transmission_gate_values[SOURCE_COMPETITION_GATE]
        )
        self.bridge.set_transmission_gate(SOURCE_COMPETITION_GATE, 0.0)
        try:
            super()._rest()
        finally:
            self.bridge.set_transmission_gate(
                SOURCE_COMPETITION_GATE, competition_gain
            )

    @staticmethod
    def _fs_region(name: str, n_neurons: int) -> BrainRegion:
        return BrainRegion(
            name=name,
            n_neurons=int(n_neurons),
            exc_fraction=0.0,
            internal_density=0.0,
            exc_weight_mean=0.0,
            inh_weight_mean=0.0,
            weight_jitter=0.0,
            plastic_internal=False,
            izh_neuron_type="IZH2007_FS_CORTICAL_INTERNEURON",
        )

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
                        plastic=False,
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

    def recall(
        self,
        episode_pattern: Sequence[int],
        *,
        source_path_lesion: bool = False,
        acc_lesion: bool = False,
    ) -> dict:
        """Recall source from episode activity and expose neural activity only."""

        episode_global = self._episode_global_indices(episode_pattern)
        self.bridge.set_plasticity_gate(SOURCE_LEARNING_GATE, 0.0)
        self.bridge.set_transmission_gate(
            SOURCE_RECALL_GATE, 0.0 if source_path_lesion else 1.0
        )
        self.bridge.set_transmission_gate(ACC_GATE, 0.0 if acc_lesion else 1.0)
        source_spikes = {source: 0.0 for source in SOURCES}
        apfc_spikes = {source: 0.0 for source in SOURCES}
        competition_spikes = {source: 0.0 for source in SOURCES}
        acc_spikes = 0.0
        try:
            self._drive(episode_global)
            for _ in range(int(self.config.read_steps)):
                self.bridge._run_one_simulation_step()
                firing = np.asarray(to_host(self.bridge.cp_firing_states), dtype=np.float64)
                for source in SOURCES:
                    source_spikes[source] += float(firing[self._source_memory_indices[source]].sum())
                    apfc_spikes[source] += float(firing[self._apfc_indices[source]].sum())
                    competition_spikes[source] += float(
                        firing[self._competition_indices[source]].sum()
                    )
                acc_spikes += float(firing[self._acc_indices].sum())
        finally:
            self._rest()
            self.bridge.set_transmission_gate(SOURCE_RECALL_GATE, 1.0)
            self.bridge.set_transmission_gate(ACC_GATE, 1.0)
            self.bridge.cp_external_input_current[:] = 0.0

        source_rates = {
            source: source_spikes[source]
            / (float(self.config.read_steps) * float(self.config.n_source_memory))
            for source in SOURCES
        }
        apfc_rates = {
            source: apfc_spikes[source]
            / (float(self.config.read_steps) * float(self.config.n_apfc))
            for source in SOURCES
        }
        return {
            "source_spikes": source_spikes,
            "source_rates": source_rates,
            "apfc_source_spikes": apfc_spikes,
            "apfc_source_rates": apfc_rates,
            "competition_spikes": competition_spikes,
            "acc_spikes": float(acc_spikes),
            "acc_rate": float(
                acc_spikes / (float(self.config.read_steps) * float(self.config.n_acc))
            ),
            "source_path_lesion": bool(source_path_lesion),
            "acc_lesion": bool(acc_lesion),
        }


def validate_calibration_seed(seed: int) -> int:
    """Open only the two v2 calibration seeds."""

    seed = int(seed)
    if seed not in CALIBRATION_SEEDS:
        raise ValueError(
            f"seed {seed} is not a v2 calibration seed; allowed={CALIBRATION_SEEDS}, "
            f"development_reserved={DEVELOPMENT_SEEDS}, held_out={HELD_OUT_SEEDS}"
        )
    return seed


def evaluate_calibration_seed(
    seed: int,
    config: SourceMonitorConfigV2 | None = None,
) -> dict:
    """Run one v2 calibration seed and all preregistered controls."""

    seed = validate_calibration_seed(seed)
    c = config or SourceMonitorConfigV2()
    patterns = make_episode_patterns(seed, 5, c)
    t0 = time.time()

    intact = SourceMonitorCoresidencyGateV2(seed=seed, config=c)
    initial = intact.weight_summary()
    intact.experience(patterns[0], visual_activity=True)
    intact.experience(patterns[1], auditory_activity=True)
    intact.experience(patterns[2], corollary_discharge=True)
    intact.experience(patterns[3], visual_activity=True, auditory_activity=True)
    learned = intact.weight_summary()
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

    swapped = SourceMonitorCoresidencyGateV2(seed=seed + 10000, config=c)
    swapped.experience(patterns[0], auditory_activity=True)
    swapped.experience(patterns[1], visual_activity=True)
    swap_zero = swapped.recall(patterns[0])
    swap_one = swapped.recall(patterns[1])

    learning_off = SourceMonitorCoresidencyGateV2(seed=seed + 20000, config=c)
    off_initial = learning_off.weight_summary()
    learning_off.experience(patterns[0], visual_activity=True, learning_enabled=False)
    off_after = learning_off.weight_summary()
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
    competition_margin_gains = {
        source: margins[source] - _source_margin(competition_lesions[source], source)
        for source in SOURCES
    }
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
        "competition_stabilizes_without_harming_source_margins": bool(
            min(competition_margin_gains.values()) >= 0.0
            and max(competition_margin_gains.values()) > 0.0
        ),
    }

    region_names = {region.name for region in intact.bridge.region_manager.regions()}
    expected_regions = {EPISODE_REGION, ACC_REGION}
    expected_regions.update(SOURCE_AFFERENT.values())
    expected_regions.update(SOURCE_MEMORY.values())
    expected_regions.update(APFC_SOURCE.values())
    expected_regions.update(SOURCE_INTERNEURON.values())
    recall_parameters = list(inspect.signature(SourceMonitorCoresidencyGateV2.recall).parameters)
    earned = Verdict("source-monitor co-residency v2 calibration")
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
        "v2 preserves v1 source input strength",
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
    earned.disabled(
        "STDP, reward modulation, homeostasis, short-term plasticity, and structural plasticity",
        why="v2 isolates Hebbian source association plus fixed local GABA-A competition",
    )
    decided = earned.decide(go=all(components.values()), verbose=False)
    status = (
        "UNDEFINED"
        if decided["status"] == UNDEFINED
        else "CALIBRATION_PASS" if decided["go"] else "CALIBRATION_FAIL"
    )
    return {
        "seed": seed,
        "phase": "calibration",
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
            not in inspect.signature(SourceMonitorCoresidencyGateV2.recall).parameters,
            "episode_activity_is_caller_supplied": True,
            "host_gain_normalization": False,
            "stronger_source_specific_drive": False,
            "host_confidence_scalar": False,
            "host_response_decision": False,
        },
        "seed_policy": {
            "calibration": list(CALIBRATION_SEEDS),
            "development_reserved": list(DEVELOPMENT_SEEDS),
            "held_out": list(HELD_OUT_SEEDS),
        },
        "fixed_criteria": {
            "minimum_source_margin": MIN_SOURCE_MARGIN,
            "minimum_attribution_fraction": MIN_ATTRIBUTION_FRACTION,
            "unseen_source_spikes": 0,
            "learning_off_source_spikes": 0,
            "all_structural_and_lesion_components_required": True,
        },
        "honest_scope": (
            "V2 adds fixed local fast-spiking GABA-A competition to the co-resident learned source circuit. "
            "Sparse episode activity, physical source-afferent identity, and the learning window remain "
            "developmental scaffolds. Population spike counts and winners are host-read for evaluation only. "
            "The competition wiring is specified rather than self-organized; no language, confidence, or "
            "response policy is claimed. The protocol gates competition off during source-free rest so "
            "post-inhibitory rebound cannot leak between trials; that timing gate remains a scaffold."
        ),
        "elapsed_seconds": round(time.time() - t0, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one calibration seed for source-monitor co-residency v2."
    )
    parser.add_argument("--seed", type=int, default=CALIBRATION_SEEDS[0])
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    row = evaluate_calibration_seed(args.seed)
    print(
        "[source-monitor-coresidency-v2] "
        f"seed={row['seed']} status={row['status']} metrics={row['metrics']} "
        f"components={row['components']}",
        flush=True,
    )
    if args.json:
        out_path = Path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(row, indent=2))
        print(f"[source-monitor-coresidency-v2] wrote {out_path}", flush=True)
    return 0 if row["status"] == "CALIBRATION_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
