"""Bounded calibration gate for co-resident episodic source monitoring.

This rung replaces hashed complete propositions with sparse activity supplied by
an upstream episode representation.  During experience, visual, auditory, or
motor-corollary afferents activate distinct source populations.  Zero-initialized
episode-to-source synapses learn that co-activity.  During recall, only episode
activity is supplied; recalled source spikes propagate to aPFC and ACC
populations on the same :class:`SimulationBridge`.

The runner is deliberately calibration-only.  Seeds 212 and 213 may be used to
tune this first bounded mechanism.  Development and held-out seeds are named
but rejected by the CLI and evaluator until the gate and thresholds are frozen.
It does not choose words, calculate confidence, or decide whether to respond.
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
from typing import Iterable, Mapping, Sequence

import numpy as np

os.environ.setdefault("SIM_BACKEND", "numpy")
logging.getLogger("SIM_BRIDGE").setLevel(logging.WARNING)

from sim.backend import get_backend, to_host
from sim.bridge import SimulationBridge
from sim.config import CoreSimConfig, GPUConfig, RuntimeState, VisualizationConfig
from sim.regions import BrainRegion, RegionPathway
from tools.lab import attributable_to


CALIBRATION_SEEDS = (212, 213)
DEVELOPMENT_SEEDS = (214, 215, 310)
HELD_OUT_SEEDS = (311, 312, 313)

SOURCES = ("seen", "heard", "self_generated")
SOURCE_AFFERENT = {
    "seen": "visual_source_afferent",
    "heard": "auditory_source_afferent",
    "self_generated": "motor_corollary_discharge",
}
SOURCE_MEMORY = {source: f"source_memory_{source}" for source in SOURCES}
APFC_SOURCE = {source: f"apfc_source_{source}" for source in SOURCES}

SOURCE_LEARNING_GATE = "episode_to_source_learning"
SOURCE_RECALL_GATE = "episode_to_source_recall"
SOURCE_AFFERENT_GATE = "source_afferent_transmission"
APFC_GATE = "source_to_apfc_transmission"
ACC_GATE = "source_to_acc_transmission"
ACC_REGION = "acc_source_monitor"
EPISODE_REGION = "episode_activity"


@dataclass(frozen=True)
class SourceMonitorConfig:
    """Small CPU-capable calibration configuration."""

    n_episode: int = 192
    episode_pattern_size: int = 12
    n_source_afferent: int = 8
    n_source_memory: int = 12
    n_apfc: int = 12
    n_acc: int = 16
    training_cycles: int = 4
    training_steps: int = 20
    rest_steps: int = 80
    read_steps: int = 100
    drive_pA: float = 5000.0
    source_afferent_weight: float = 80.0
    source_to_apfc_weight: float = 12.0
    source_to_acc_weight: float = 8.0
    hebbian_learning_rate: float = 0.2
    hebbian_max_weight: float = 160.0


def make_episode_patterns(
    seed: int,
    n_patterns: int,
    config: SourceMonitorConfig | None = None,
) -> list[np.ndarray]:
    """Return disjoint sparse episode activity for a bounded synthetic world.

    The generated arrays stand in for upstream perceptual or replay activity;
    they are passed to the bridge directly and are never reconstructed from a
    proposition, answer string, or source label.
    """

    c = config or SourceMonitorConfig()
    needed = int(n_patterns) * int(c.episode_pattern_size)
    if needed > int(c.n_episode):
        raise ValueError("requested disjoint patterns exceed the episode population")
    order = np.random.default_rng(int(seed)).permutation(int(c.n_episode))[:needed]
    return [
        np.sort(order[i : i + int(c.episode_pattern_size)]).astype(np.int64)
        for i in range(0, needed, int(c.episode_pattern_size))
    ]


class SourceMonitorCoresidencyGate:
    """One bridge containing episode, source-memory, aPFC, and ACC populations."""

    def __init__(
        self,
        *,
        seed: int,
        config: SourceMonitorConfig | Mapping | None = None,
    ):
        self.seed = int(seed)
        self.config = (
            config
            if isinstance(config, SourceMonitorConfig)
            else SourceMonitorConfig(**(dict(config) if config else {}))
        )
        self._validate_config()
        self.bridge = self._build_bridge()
        rm = self.bridge.region_manager
        self._episode_indices = np.asarray(rm.indices(EPISODE_REGION), dtype=np.int64)
        self._source_afferent_indices = {
            source: np.asarray(rm.indices(SOURCE_AFFERENT[source]), dtype=np.int64)
            for source in SOURCES
        }
        self._source_memory_indices = {
            source: np.asarray(rm.indices(SOURCE_MEMORY[source]), dtype=np.int64)
            for source in SOURCES
        }
        self._apfc_indices = {
            source: np.asarray(rm.indices(APFC_SOURCE[source]), dtype=np.int64)
            for source in SOURCES
        }
        self._acc_indices = np.asarray(rm.indices(ACC_REGION), dtype=np.int64)
        self.bridge.set_plasticity_gate(SOURCE_LEARNING_GATE, 0.0)
        for gate in (SOURCE_RECALL_GATE, SOURCE_AFFERENT_GATE, APFC_GATE, ACC_GATE):
            self.bridge.set_transmission_gate(gate, 1.0)
        self._zero_learned_weights()

    def _validate_config(self) -> None:
        c = self.config
        if c.n_episode < 1 or c.episode_pattern_size < 1:
            raise ValueError("episode population and pattern size must be positive")
        if c.episode_pattern_size > c.n_episode:
            raise ValueError("episode_pattern_size must fit inside n_episode")
        for name in ("n_source_afferent", "n_source_memory", "n_apfc", "n_acc"):
            if int(getattr(c, name)) < 1:
                raise ValueError(f"{name} must be positive")

    @staticmethod
    def _region(name: str, n_neurons: int) -> BrainRegion:
        return BrainRegion(
            name=name,
            n_neurons=int(n_neurons),
            exc_fraction=1.0,
            internal_density=0.0,
            exc_weight_mean=0.0,
            inh_weight_mean=0.0,
            weight_jitter=0.0,
            plastic_internal=False,
        )

    def _build_bridge(self) -> SimulationBridge:
        c = self.config
        regions = [self._region(EPISODE_REGION, c.n_episode)]
        for source in SOURCES:
            regions.extend(
                [
                    self._region(SOURCE_AFFERENT[source], c.n_source_afferent),
                    self._region(SOURCE_MEMORY[source], c.n_source_memory),
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

    def _learned_synapse_indices(self) -> np.ndarray:
        return np.asarray(
            self.bridge._plasticity_gate_to_synapses[SOURCE_LEARNING_GATE],
            dtype=np.int64,
        )

    def _zero_learned_weights(self) -> None:
        xp, _ = get_backend()
        indices = xp.asarray(self._learned_synapse_indices(), dtype=xp.int64)
        self.bridge.cp_connections.data[indices] = 0.0

    def weight_summary(self) -> dict[str, float | int]:
        data = np.asarray(to_host(self.bridge.cp_connections.data), dtype=np.float64)
        weights = data[self._learned_synapse_indices()]
        return {
            "n_synapses": int(weights.size),
            "nonzero": int(np.count_nonzero(np.abs(weights) > 1e-8)),
            "mean": float(weights.mean()) if weights.size else 0.0,
            "max": float(weights.max()) if weights.size else 0.0,
            "l1": float(np.abs(weights).sum()),
        }

    def _episode_global_indices(self, episode_pattern: Sequence[int]) -> np.ndarray:
        local = np.asarray(episode_pattern, dtype=np.int64)
        if local.ndim != 1 or local.size == 0:
            raise ValueError("episode_pattern must be a non-empty one-dimensional index sequence")
        if np.unique(local).size != local.size:
            raise ValueError("episode_pattern indices must be unique")
        if np.any(local < 0) or np.any(local >= int(self.config.n_episode)):
            raise ValueError("episode_pattern index is outside the episode population")
        return self._episode_indices[local]

    @staticmethod
    def _active_sources(
        *,
        visual_activity: bool,
        auditory_activity: bool,
        corollary_discharge: bool,
    ) -> tuple[str, ...]:
        active = []
        if visual_activity:
            active.append("seen")
        if auditory_activity:
            active.append("heard")
        if corollary_discharge:
            active.append("self_generated")
        if not active:
            raise ValueError("experience requires sensory activity or corollary discharge")
        return tuple(active)

    def _drive(self, episode_global: np.ndarray, active_sources: Iterable[str] = ()) -> None:
        xp, _ = get_backend()
        self.bridge.cp_external_input_current[:] = 0.0
        self.bridge.cp_external_input_current[
            xp.asarray(episode_global, dtype=xp.int64)
        ] = float(self.config.drive_pA)
        for source in active_sources:
            self.bridge.cp_external_input_current[
                xp.asarray(self._source_afferent_indices[source], dtype=xp.int64)
            ] = float(self.config.drive_pA)

    def _rest(self) -> None:
        self.bridge.cp_external_input_current[:] = 0.0
        for _ in range(int(self.config.rest_steps)):
            self.bridge._run_one_simulation_step()

    def experience(
        self,
        episode_pattern: Sequence[int],
        *,
        visual_activity: bool = False,
        auditory_activity: bool = False,
        corollary_discharge: bool = False,
        learning_enabled: bool = True,
        source_afferent_lesion: bool = False,
    ) -> dict:
        """Co-activate episode and physical source pathways for one experience."""

        episode_global = self._episode_global_indices(episode_pattern)
        active_sources = self._active_sources(
            visual_activity=visual_activity,
            auditory_activity=auditory_activity,
            corollary_discharge=corollary_discharge,
        )
        before = self.weight_summary()
        self.bridge.set_plasticity_gate(
            SOURCE_LEARNING_GATE,
            1.0 if learning_enabled else 0.0,
        )
        self.bridge.set_transmission_gate(
            SOURCE_AFFERENT_GATE,
            0.0 if source_afferent_lesion else 1.0,
        )
        try:
            for _ in range(int(self.config.training_cycles)):
                self._drive(episode_global, active_sources)
                for _ in range(int(self.config.training_steps)):
                    self.bridge._run_one_simulation_step()
                self._rest()
        finally:
            self.bridge.set_plasticity_gate(SOURCE_LEARNING_GATE, 0.0)
            self.bridge.set_transmission_gate(SOURCE_AFFERENT_GATE, 1.0)
            self.bridge.cp_external_input_current[:] = 0.0
        after = self.weight_summary()
        return {
            "active_afferents": list(active_sources),
            "learning_enabled": bool(learning_enabled),
            "source_afferent_lesion": bool(source_afferent_lesion),
            "weight_l1_before": float(before["l1"]),
            "weight_l1_after": float(after["l1"]),
            "weight_l1_delta": float(after["l1"] - before["l1"]),
        }

    def recall(
        self,
        episode_pattern: Sequence[int],
        *,
        source_path_lesion: bool = False,
        acc_lesion: bool = False,
    ) -> dict:
        """Drive episode activity alone and measure downstream neural populations."""

        episode_global = self._episode_global_indices(episode_pattern)
        self.bridge.set_plasticity_gate(SOURCE_LEARNING_GATE, 0.0)
        self.bridge.set_transmission_gate(
            SOURCE_RECALL_GATE,
            0.0 if source_path_lesion else 1.0,
        )
        self.bridge.set_transmission_gate(ACC_GATE, 0.0 if acc_lesion else 1.0)
        source_spikes = {source: 0.0 for source in SOURCES}
        apfc_spikes = {source: 0.0 for source in SOURCES}
        acc_spikes = 0.0
        try:
            self._drive(episode_global)
            for _ in range(int(self.config.read_steps)):
                self.bridge._run_one_simulation_step()
                firing = np.asarray(to_host(self.bridge.cp_firing_states), dtype=np.float64)
                for source in SOURCES:
                    source_spikes[source] += float(firing[self._source_memory_indices[source]].sum())
                    apfc_spikes[source] += float(firing[self._apfc_indices[source]].sum())
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
            "acc_spikes": float(acc_spikes),
            "acc_rate": float(
                acc_spikes / (float(self.config.read_steps) * float(self.config.n_acc))
            ),
            "source_path_lesion": bool(source_path_lesion),
            "acc_lesion": bool(acc_lesion),
        }


def _dominant_source(record: Mapping) -> str:
    rates = record["source_rates"]
    return max(SOURCES, key=lambda source: float(rates[source]))


def _source_margin(record: Mapping, expected: str) -> float:
    rates = record["source_rates"]
    alternatives = [float(rates[source]) for source in SOURCES if source != expected]
    return float(rates[expected]) - max(alternatives)


def evaluate_calibration_seed(
    seed: int,
    config: SourceMonitorConfig | None = None,
) -> dict:
    """Run the bounded controls without consuming development or held-out seeds."""

    if int(seed) not in CALIBRATION_SEEDS:
        raise ValueError(
            f"seed {seed} is not a calibration seed; allowed={CALIBRATION_SEEDS}, "
            f"development_reserved={DEVELOPMENT_SEEDS}, held_out={HELD_OUT_SEEDS}"
        )
    c = config or SourceMonitorConfig()
    patterns = make_episode_patterns(seed, 4, c)
    t0 = time.time()

    intact = SourceMonitorCoresidencyGate(seed=seed, config=c)
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
    source_lesion = intact.recall(patterns[0], source_path_lesion=True)
    acc_lesion = intact.recall(patterns[0], acc_lesion=True)

    swapped = SourceMonitorCoresidencyGate(seed=seed + 10000, config=c)
    swapped.experience(patterns[0], auditory_activity=True)
    swapped.experience(patterns[1], visual_activity=True)
    swap_zero = swapped.recall(patterns[0])
    swap_one = swapped.recall(patterns[1])

    learning_off = SourceMonitorCoresidencyGate(seed=seed + 20000, config=c)
    off_initial = learning_off.weight_summary()
    learning_off.experience(patterns[0], visual_activity=True, learning_enabled=False)
    off_after = learning_off.weight_summary()
    off_recall = learning_off.recall(patterns[0])

    intact_source_total = float(sum(seen["source_spikes"].values()))
    lesioned_source_total = float(sum(source_lesion["source_spikes"].values()))
    source_path_fraction = attributable_to(
        "source recall pathway", intact_source_total, lesioned_source_total,
    )
    acc_path_fraction = attributable_to(
        "source-to-ACC pathway",
        float(seen["acc_spikes"]),
        float(acc_lesion["acc_spikes"]),
    )
    components = {
        "learned_routes_start_zero": bool(initial["l1"] == 0.0),
        "experience_changes_synaptic_weights": bool(learned["l1"] > initial["l1"]),
        "seen_source_recalled": bool(_dominant_source(seen) == "seen" and _source_margin(seen, "seen") > 0.0),
        "heard_source_recalled": bool(_dominant_source(heard) == "heard" and _source_margin(heard, "heard") > 0.0),
        "self_source_recalled": bool(
            _dominant_source(self_generated) == "self_generated"
            and _source_margin(self_generated, "self_generated") > 0.0
        ),
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
        "acc_lesion_preserves_source_and_silences_acc": bool(
            sum(acc_lesion["source_spikes"].values()) >= 0.90 * intact_source_total
            and acc_lesion["acc_spikes"] == 0.0
        ),
        "learning_off_keeps_weights_zero": bool(
            off_initial["l1"] == 0.0 and off_after["l1"] == 0.0
        ),
        "learning_off_has_no_source_recall": bool(sum(off_recall["source_spikes"].values()) == 0.0),
        "source_spikes_reach_apfc_and_acc": bool(
            seen["apfc_source_spikes"]["seen"] > 0.0 and seen["acc_spikes"] > 0.0
        ),
    }
    return {
        "seed": int(seed),
        "status": "CALIBRATION_PASS" if all(components.values()) else "NEEDS_TUNING",
        "components": components,
        "metrics": {
            "seen_margin": _source_margin(seen, "seen"),
            "heard_margin": _source_margin(heard, "heard"),
            "self_generated_margin": _source_margin(self_generated, "self_generated"),
            "intact_source_spikes": intact_source_total,
            "source_lesion_spikes": lesioned_source_total,
            "intact_acc_spikes": float(seen["acc_spikes"]),
            "acc_lesion_spikes": float(acc_lesion["acc_spikes"]),
            "mixed_seen_spikes": float(mixed["source_spikes"]["seen"]),
            "mixed_heard_spikes": float(mixed["source_spikes"]["heard"]),
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
            "source_path_lesion": source_lesion,
            "acc_lesion": acc_lesion,
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
        "interface_guards": {
            "recall_parameters": list(inspect.signature(SourceMonitorCoresidencyGate.recall).parameters),
            "no_source_argument_at_inference": "source" not in inspect.signature(
                SourceMonitorCoresidencyGate.recall
            ).parameters,
            "episode_activity_is_caller_supplied": True,
            "host_confidence_scalar": False,
            "host_response_decision": False,
        },
        "seed_policy": {
            "calibration": list(CALIBRATION_SEEDS),
            "development_reserved": list(DEVELOPMENT_SEEDS),
            "held_out": list(HELD_OUT_SEEDS),
        },
        "honest_scope": (
            "The co-resident bridge learns source from sparse episode activity plus physical visual, auditory, "
            "or corollary-discharge afferents and propagates recall into neural aPFC/ACC populations. Sparse "
            "episode activity is supplied by the test world; source afferent identity and the plasticity window "
            "remain developmental scaffolds. Population counts are host-read for evaluation only. There is no "
            "language integration, confidence scalar, speech policy, or claim that source monitoring is solved."
        ),
        "elapsed_seconds": round(time.time() - t0, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one bounded source-monitor co-residency calibration seed."
    )
    parser.add_argument("--seed", type=int, choices=CALIBRATION_SEEDS, default=CALIBRATION_SEEDS[0])
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    row = evaluate_calibration_seed(int(args.seed))
    print(
        "[source-monitor-coresidency] "
        f"seed={row['seed']} status={row['status']} components={row['components']}",
        flush=True,
    )
    if args.json:
        out_path = Path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(row, indent=2))
        print(f"[source-monitor-coresidency] wrote {out_path}", flush=True)
    return 0 if row["status"] == "CALIBRATION_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
