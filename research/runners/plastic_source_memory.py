"""Plastic episodic source support for Lane C honesty.

The memory learns whether a complete proposition was externally experienced. A
deterministic sparse proposition code drives several independent source
populations through zero-initialized Hebbian pathways. Experience co-activates
the proposition and source populations while the learning gate is open.
Retrieval drives only the live proposition and reads source spikes.

This is an opt-in research rung. The sparse code construction and the separate
bridge remain scaffolds; the proposition-to-source association itself lives in
learned synaptic weights and no expected answer is retained for inference.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from sim.backend import get_backend, to_host
from sim.bridge import SimulationBridge
from sim.config import CoreSimConfig, GPUConfig, RuntimeState, VisualizationConfig
from sim.regions import BrainRegion, RegionPathway


LEARNING_GATE = "plastic_source_learn"
TRANSMISSION_GATE = "plastic_source_read"


@dataclass(frozen=True)
class PlasticSourceConfig:
    n_banks: int = 4
    proposition_neurons_per_bank: int = 16384
    proposition_pattern_size: int = 16
    source_neurons_per_bank: int = 8
    training_cycles: int = 3
    training_steps: int = 20
    rest_steps: int = 10
    read_steps: int = 80
    drive_pA: float = 5000.0
    support_threshold: float = 0.34
    hebbian_learning_rate: float = 0.2
    hebbian_max_weight: float = 160.0


class PlasticSourceMemory:
    """Learned proposition-to-source association on a spiking bridge."""

    def __init__(
        self,
        *,
        seed: int,
        config: PlasticSourceConfig | Mapping | None = None,
    ):
        self.seed = int(seed)
        self.config = (
            config
            if isinstance(config, PlasticSourceConfig)
            else PlasticSourceConfig(**(dict(config) if config else {}))
        )
        if self.config.n_banks < 1:
            raise ValueError("n_banks must be positive")
        if not 0 < self.config.proposition_pattern_size <= self.config.proposition_neurons_per_bank:
            raise ValueError("proposition_pattern_size must fit within each proposition bank")
        self.bridge = self._build_bridge()
        rm = self.bridge.region_manager
        self._proposition_indices = [
            np.asarray(rm.indices(f"source_proposition_{i}"), dtype=np.int64)
            for i in range(self.config.n_banks)
        ]
        self._source_indices = [
            np.asarray(rm.indices(f"source_external_{i}"), dtype=np.int64)
            for i in range(self.config.n_banks)
        ]
        self._observed_events = 0
        self.bridge.set_plasticity_gate(LEARNING_GATE, 0.0)
        self.bridge.set_transmission_gate(TRANSMISSION_GATE, 1.0)
        self._zero_source_weights()
        self._liveness_spikes = self._measure_source_liveness()
        if np.any(self._liveness_spikes <= 0.0):
            raise RuntimeError("plastic source populations failed their direct-drive liveness check")

    def _build_bridge(self) -> SimulationBridge:
        c = self.config
        regions = []
        pathways = []
        for bank in range(c.n_banks):
            regions.extend([
                BrainRegion(
                    name=f"source_proposition_{bank}",
                    n_neurons=c.proposition_neurons_per_bank,
                    exc_fraction=1.0,
                    internal_density=0.0,
                    exc_weight_mean=0.0,
                    inh_weight_mean=0.0,
                    weight_jitter=0.0,
                    plastic_internal=False,
                ),
                BrainRegion(
                    name=f"source_external_{bank}",
                    n_neurons=c.source_neurons_per_bank,
                    exc_fraction=1.0,
                    internal_density=0.0,
                    exc_weight_mean=0.0,
                    inh_weight_mean=0.0,
                    weight_jitter=0.0,
                    plastic_internal=False,
                ),
            ])
            pathways.append(RegionPathway(
                from_region=f"source_proposition_{bank}",
                to_region=f"source_external_{bank}",
                density=1.0,
                weight_mean=0.0,
                weight_jitter=0.0,
                plastic=True,
                plasticity_gate=LEARNING_GATE,
                transmission_gate=TRANSMISSION_GATE,
            ))

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

    def _source_synapse_indices(self):
        return np.asarray(
            self.bridge._plasticity_gate_to_synapses[LEARNING_GATE],
            dtype=np.int64,
        )

    def _zero_source_weights(self):
        xp, _ = get_backend()
        idx = xp.asarray(self._source_synapse_indices(), dtype=xp.int64)
        self.bridge.cp_connections.data[idx] = 0.0

    @staticmethod
    def _canonical_payload(kind: str, cue: Sequence[str], candidate: str) -> bytes:
        return json.dumps(
            [str(kind), [str(x) for x in cue], str(candidate)],
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")

    def _proposition_pattern(self, *, kind: str, cue: Sequence[str], candidate: str) -> np.ndarray:
        payload = self._canonical_payload(kind, cue, candidate)
        out = []
        c = self.config
        for bank, base in enumerate(self._proposition_indices):
            digest = hashlib.blake2b(
                payload,
                digest_size=16,
                person=f"src-bank-{bank}".encode("ascii"),
            ).digest()
            pattern_seed = int.from_bytes(digest[:8], "little") ^ self.seed
            rng = np.random.default_rng(pattern_seed)
            local = rng.choice(
                c.proposition_neurons_per_bank,
                c.proposition_pattern_size,
                replace=False,
            )
            out.extend(base[local])
        return np.asarray(out, dtype=np.int64)

    def _all_source_indices(self) -> np.ndarray:
        return np.concatenate(self._source_indices)

    def _drive(self, indices: Iterable[int], *, include_source=False):
        xp, _ = get_backend()
        ext = xp.zeros_like(self.bridge.cp_external_input_current)
        idx = np.asarray(list(indices), dtype=np.int64)
        if idx.size:
            ext[xp.asarray(idx, dtype=xp.int64)] = float(self.config.drive_pA)
        if include_source:
            src = self._all_source_indices()
            ext[xp.asarray(src, dtype=xp.int64)] = float(self.config.drive_pA)
        self.bridge.cp_external_input_current[:] = ext

    def _rest(self, steps: int | None = None):
        self.bridge.cp_external_input_current[:] = 0.0
        for _ in range(int(self.config.rest_steps if steps is None else steps)):
            self.bridge._run_one_simulation_step()

    def _read_source_spikes(self, proposition_pattern: np.ndarray, *, direct=False) -> np.ndarray:
        self._drive(proposition_pattern, include_source=direct)
        counts = np.zeros(self.config.n_banks, dtype=np.float64)
        for _ in range(int(self.config.read_steps)):
            self.bridge._run_one_simulation_step()
            firing = np.asarray(to_host(self.bridge.cp_firing_states), dtype=np.float64)
            for bank, idx in enumerate(self._source_indices):
                counts[bank] += float(firing[idx].sum())
        self._rest()
        return counts

    def _measure_source_liveness(self) -> np.ndarray:
        return self._read_source_spikes(np.asarray([], dtype=np.int64), direct=True)

    def weight_summary(self) -> dict:
        data = np.asarray(to_host(self.bridge.cp_connections.data), dtype=np.float64)
        weights = data[self._source_synapse_indices()]
        return {
            "n_synapses": int(weights.size),
            "nonzero": int(np.count_nonzero(np.abs(weights) > 1e-8)),
            "mean": float(weights.mean()) if weights.size else 0.0,
            "max": float(weights.max()) if weights.size else 0.0,
            "l1": float(np.abs(weights).sum()),
        }

    def observe(
        self,
        *,
        kind: str,
        cue: Sequence[str],
        candidate: str,
        learning_enabled=True,
        measure_weights=False,
    ) -> dict:
        """Present one source event; content is discarded after synaptic learning."""
        pattern = self._proposition_pattern(kind=kind, cue=cue, candidate=candidate)
        before = self.weight_summary() if measure_weights else None
        self._observed_events += 1
        self.bridge.set_plasticity_gate(LEARNING_GATE, 1.0 if learning_enabled else 0.0)
        try:
            for _ in range(int(self.config.training_cycles)):
                self._drive(pattern, include_source=True)
                for _ in range(int(self.config.training_steps)):
                    self.bridge._run_one_simulation_step()
                self._rest()
        finally:
            self.bridge.set_plasticity_gate(LEARNING_GATE, 0.0)
        after = self.weight_summary() if measure_weights else None
        return {
            "learning_enabled": bool(learning_enabled),
            "weight_l1_before": before["l1"] if before is not None else None,
            "weight_l1_after": after["l1"] if after is not None else None,
            "weight_l1_delta": (
                float(after["l1"] - before["l1"])
                if before is not None and after is not None
                else None
            ),
        }

    def support(self, *, kind: str, cue: Sequence[str], candidate: str, lesion=False) -> dict:
        """Read spike support for the live candidate without an expected answer."""
        self.bridge.set_plasticity_gate(LEARNING_GATE, 0.0)
        self.bridge.set_transmission_gate(TRANSMISSION_GATE, 0.0 if lesion else 1.0)
        try:
            pattern = self._proposition_pattern(kind=kind, cue=cue, candidate=candidate)
            spikes = self._read_source_spikes(pattern)
        finally:
            self.bridge.set_transmission_gate(TRANSMISSION_GATE, 1.0)
        bank_support = spikes / np.maximum(self._liveness_spikes, 1.0)
        support = float(np.mean(bank_support))
        available = self._observed_events > 0
        consistent = bool(support >= self.config.support_threshold) if available else None
        return {
            "available": bool(available),
            "source": "plastic_hebbian_proposition_source",
            "source_consistent": consistent,
            "source_confidence": float(np.clip(support, 0.0, 1.0)),
            "support": support,
            "support_threshold": float(self.config.support_threshold),
            "bank_support": [float(x) for x in bank_support],
            "source_spikes": [float(x) for x in spikes],
            "source_liveness_spikes": [float(x) for x in self._liveness_spikes],
            "observed_events": int(self._observed_events),
            "lesion": bool(lesion),
            "config": asdict(self.config),
        }
