"""Lane C self-schema honesty hook for production conversation.

This module is intentionally small and default-off. It consumes an answer-process
confidence scalar from the conversation path, drives a metacognitive confidence
population, then reads a downstream self_schema confidence pool through fixed
synapses. The resulting self-schema firing rate can only downgrade a matched
answer into a hedge or soft abstain; it never turns a hard moat miss into an
answer.

Scope: this is a production wire-in of the Lane C self-schema relay, not a new
claim of subjective experience.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np

from sim import GPUConfig, RuntimeState, SimulationBridge, VisualizationConfig
from sim.backend import get_backend, to_host
from sim.config import CoreSimConfig
from sim.enums import NeuronModel
from research.runners._gnw_rung1_ignition_curve_derisk import _restore_state, _snapshot_state
from tools.lab import attributable_to


CONFIDENCE_SOURCE_TRACE = "trace"
CONFIDENCE_SOURCE_SOURCE_CONSISTENCY_FLOOR = "source_consistency_floor"
CONFIDENCE_SOURCE_NEURAL_SOURCE_CONSISTENCY = "neural_source_consistency"
CONFIDENCE_SOURCE_PLASTIC_SOURCE_CONSISTENCY = "plastic_source_consistency"
CONFIDENCE_SOURCE_CHOICES = (
    CONFIDENCE_SOURCE_TRACE,
    CONFIDENCE_SOURCE_SOURCE_CONSISTENCY_FLOOR,
    CONFIDENCE_SOURCE_NEURAL_SOURCE_CONSISTENCY,
    CONFIDENCE_SOURCE_PLASTIC_SOURCE_CONSISTENCY,
)


@dataclass(frozen=True)
class SelfSchemaHonestyConfig:
    """Operating point for the small production self-schema confidence readout."""

    n_meta: int = 24
    n_self: int = 24
    meta_to_self_w: float = 20.0
    max_meta_current_pa: float = 800.0
    settle_steps: int = 20
    report_steps: int = 60
    confidence_assert: float = 0.55
    confidence_hedge: float = 0.38
    require_source_floor: bool = True
    confidence_source_mode: str = CONFIDENCE_SOURCE_TRACE


def self_schema_hedge_text(kind: str, answer: Any, *, cue: tuple[Any, ...] | None = None) -> str:
    """Render a conservative hedge without introducing new content."""
    if kind == "what_does":
        ag, ac = cue[:2] if cue is not None else ("that", "does")
        return f"I am not certain, but {ag} {ac} {answer}."
    if kind == "yes_no":
        return f"I am not certain, but {answer}."
    return f"I am not certain, but {answer}."


def self_schema_soft_abstain_text(kind: str, answer: Any, *, cue: tuple[Any, ...] | None = None) -> str:
    """Render a soft abstain that surfaces only the already-recalled candidate."""
    if kind == "what_does":
        return f"I am not sure enough to assert it, but it might be {answer}."
    if kind == "yes_no":
        return "I am not sure enough to answer yes or no."
    return "I am not sure enough to assert that."


def trace_confidence(last_trace: Mapping[str, Any] | None, preferred_role: str = "patient") -> float | None:
    """Extract a query confidence scalar from a composer trace.

    For what-does queries, the preferred non-cue patient chip is the useful
    answer-process confidence. For yes/no traces, the patient is a cue, so the
    fallback is the minimum non-null cue confidence over the matched fact.
    """
    if not last_trace:
        return None
    roles = list(last_trace.get("roles", []))
    for ch in roles:
        if ch.get("role") == preferred_role and not ch.get("cue", False):
            c = ch.get("confidence")
            return None if c is None else float(c)
    cue_conf = [
        float(ch["confidence"])
        for ch in roles
        if ch.get("cue", False) and ch.get("confidence") is not None
    ]
    if cue_conf:
        return float(min(cue_conf))
    any_conf = [
        float(ch["confidence"])
        for ch in roles
        if ch.get("confidence") is not None
    ]
    return float(min(any_conf)) if any_conf else None


def recall_trace_evidence(
    last_trace: Mapping[str, Any] | None,
    *,
    preferred_role: str = "patient",
) -> dict[str, Any]:
    """Extract read-only evidence from a composer recall trace.

    The fields are the recall process's own cleanup/conflict values. `source_fact`
    is optional trace metadata from composer scaffolds and must be treated as a
    source-provenance scaffold, not as a final biological mechanism.
    """
    raw_conf = trace_confidence(last_trace, preferred_role=preferred_role)
    if not last_trace:
        return {
            "raw_trace_confidence": raw_conf,
            "answer_confidence": None,
            "answer_margin": None,
            "answer_conflict": None,
            "cue_min_confidence": None,
            "cue_min_margin": None,
            "matched_fact_index": None,
            "source_fact": None,
        }
    roles = list(last_trace.get("roles", []))
    answer = None
    for ch in roles:
        if ch.get("role") == preferred_role and not ch.get("cue", False):
            answer = ch
            break
    if answer is None:
        for ch in roles:
            if not ch.get("cue", False):
                answer = ch
                break
    cue_conf = [
        float(ch["confidence"])
        for ch in roles
        if ch.get("cue", False) and ch.get("confidence") is not None
    ]
    cue_margin = [
        float(ch["margin"])
        for ch in roles
        if ch.get("cue", False) and ch.get("margin") is not None
    ]
    return {
        "raw_trace_confidence": raw_conf,
        "answer_confidence": (
            None if answer is None or answer.get("confidence") is None else float(answer["confidence"])
        ),
        "answer_margin": None if answer is None or answer.get("margin") is None else float(answer["margin"]),
        "answer_conflict": None if answer is None or answer.get("conflict") is None else float(answer["conflict"]),
        "cue_min_confidence": float(min(cue_conf)) if cue_conf else None,
        "cue_min_margin": float(min(cue_margin)) if cue_margin else None,
        "matched_fact_index": last_trace.get("matched_fact_index"),
        "source_fact": last_trace.get("source_fact"),
    }


def _source_expected_answer(kind: str, source_fact: Mapping[str, Any] | None) -> Any:
    if not source_fact:
        return None
    if kind == "what_does":
        patient = source_fact.get("patient")
        attrs = [
            source_fact[r]
            for r in ("attribute", "attribute2")
            if r in source_fact and source_fact.get(r) is not None
        ]
        return " ".join([str(x) for x in attrs + [patient]]) if attrs else patient
    if kind == "yes_no":
        return "no" if source_fact.get("polarity") == "NEGATE" else "yes"
    return None


def _source_cue_matches(kind: str, cue: tuple[Any, ...], source_fact: Mapping[str, Any] | None) -> bool | None:
    if not source_fact:
        return None
    if kind == "what_does" and len(cue) == 2:
        ag, ac = cue
        return bool(source_fact.get("agent") == ag and source_fact.get("action") == ac)
    if kind == "yes_no" and len(cue) == 3:
        ag, ac, pt = cue
        return bool(
            source_fact.get("agent") == ag
            and source_fact.get("action") == ac
            and source_fact.get("patient") == pt
        )
    return None


def known_fact_confidence_record(
    last_trace: Mapping[str, Any] | None,
    *,
    kind: str,
    cue: tuple[Any, ...],
    raw_answer: Any,
    mode: str = CONFIDENCE_SOURCE_TRACE,
    source_monitor_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Choose the confidence scalar used by the self-schema relay.

    Default mode is the previous raw trace confidence. `source_consistency_floor`
    is an explicitly named scaffold: it lets an exact source fact attached to the
    trace veto a cleanup-decoded answer that disagrees with that source record.
    It is useful as a production safety floor but should be burned down into a
    neural source-memory/readout consistency signal.
    """
    if mode not in CONFIDENCE_SOURCE_CHOICES:
        raise ValueError(f"unknown Lane C confidence_source_mode={mode!r}")
    evidence = recall_trace_evidence(last_trace)
    confidence = evidence["raw_trace_confidence"]
    source_answer = _source_expected_answer(kind, evidence.get("source_fact"))
    if source_answer is None:
        source_matches = None
    else:
        source_matches = bool(source_answer == raw_answer)
    source_cue_matches = _source_cue_matches(kind, tuple(cue), evidence.get("source_fact"))
    source_consistent = None
    if source_matches is not None or source_cue_matches is not None:
        source_consistent = bool(source_matches is not False and source_cue_matches is not False)
    exact_source_consistent = source_consistent
    source_monitor = dict(source_monitor_evidence or {})
    neural_source_consistent = source_monitor.get("source_consistent")
    selected_consistency_source = "trace_source_fact"
    if mode in (
        CONFIDENCE_SOURCE_NEURAL_SOURCE_CONSISTENCY,
        CONFIDENCE_SOURCE_PLASTIC_SOURCE_CONSISTENCY,
    ):
        selected_consistency_source = source_monitor.get(
            "source",
            "rf_independent_source_echo"
            if mode == CONFIDENCE_SOURCE_NEURAL_SOURCE_CONSISTENCY
            else "plastic_hebbian_proposition_source",
        )
        if not source_monitor.get("available", False):
            source_consistent = False
        else:
            source_consistent = neural_source_consistent if neural_source_consistent is not None else False
    if mode in (
        CONFIDENCE_SOURCE_SOURCE_CONSISTENCY_FLOOR,
        CONFIDENCE_SOURCE_NEURAL_SOURCE_CONSISTENCY,
        CONFIDENCE_SOURCE_PLASTIC_SOURCE_CONSISTENCY,
    ) and source_consistent is False:
        confidence = 0.0
    evidence.update({
        "mode": mode,
        "selected_confidence": None if confidence is None else float(confidence),
        "source_expected_answer": source_answer,
        "source_answer_matches": source_matches,
        "source_cue_matches": source_cue_matches,
        "exact_source_consistent": exact_source_consistent,
        "neural_source_monitor": source_monitor,
        "neural_source_consistent": neural_source_consistent,
        "selected_consistency_source": selected_consistency_source,
        "source_consistent": source_consistent,
        "scaffold": bool(mode == CONFIDENCE_SOURCE_SOURCE_CONSISTENCY_FLOOR),
        "learned_source_association": bool(
            mode == CONFIDENCE_SOURCE_PLASTIC_SOURCE_CONSISTENCY
        ),
    })
    return evidence


def self_read_attribution(intact_rate: float, lesioned_rate: float) -> float | None:
    """Attribute a self-schema rate to the meta->self read projection."""
    return attributable_to(
        "laneC self_schema self-read relay",
        float(intact_rate),
        float(lesioned_rate),
        warn_below=0.5,
    )


class SelfSchemaHonestyMonitor:
    """Tiny fixed meta_schema -> self_schema confidence relay."""

    def __init__(
        self,
        seed: int = 42,
        config: SelfSchemaHonestyConfig | Mapping[str, Any] | None = None,
        *,
        lesion_self_read: bool = False,
    ):
        self.seed = int(seed)
        self.config = (
            config
            if isinstance(config, SelfSchemaHonestyConfig)
            else SelfSchemaHonestyConfig(**(dict(config) if config else {}))
        )
        self.lesion_self_read = bool(lesion_self_read)
        self._build_bridge()
        self._assert_rate = self._run_rate(self.config.confidence_assert)
        self._hedge_rate = self._run_rate(self.config.confidence_hedge)
        if self._hedge_rate > self._assert_rate:
            self._hedge_rate, self._assert_rate = self._assert_rate, self._hedge_rate

    def _build_bridge(self) -> None:
        xp, _ = get_backend()
        cfg = CoreSimConfig()
        cfg.num_neurons = int(self.config.n_meta + self.config.n_self)
        cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
        cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
        cfg.seed = self.seed
        cfg.dt_ms = 1.0
        cfg.connections_per_neuron = 0
        cfg.num_traits = 1
        cfg.ou_std_current_pA = 0.0
        cfg.enable_parameter_heterogeneity = True
        for flag in (
            "enable_stdp",
            "enable_hebbian_learning",
            "enable_short_term_plasticity",
            "enable_structural_plasticity",
            "enable_homeostasis",
            "enable_reward_modulation",
            "enable_watts_strogatz",
            "enable_neuromodulator_subsystem",
            "enable_brain_region_framework",
            "enable_ou_process",
        ):
            if hasattr(cfg, flag):
                setattr(cfg, flag, False)

        bridge = SimulationBridge(
            core_config=cfg,
            viz_config=VisualizationConfig(),
            runtime_state=RuntimeState(),
            gpu_config=GPUConfig(),
        )
        bridge._initialize_simulation_data(called_from_playback_init=False)

        meta = np.arange(int(self.config.n_meta), dtype=np.int64)
        self_idx = np.arange(
            int(self.config.n_meta),
            int(self.config.n_meta + self.config.n_self),
            dtype=np.int64,
        )
        pre = np.repeat(meta, len(self_idx))
        post = np.tile(self_idx, len(meta))
        w = 0.0 if self.lesion_self_read else float(self.config.meta_to_self_w)
        bridge.inject_explicit_wiring(
            {
                "laneC_meta_to_self_confidence": {
                    "pre_indices": pre,
                    "post_indices": post,
                    "initial_weights": np.full(pre.shape[0], w, dtype=np.float32),
                    "plastic": False,
                    "conn_type": "E_TO_E",
                    "count": int(pre.shape[0]),
                }
            }
        )
        bridge.cp_external_input_current[:] = 0.0
        for _ in range(int(self.config.settle_steps)):
            bridge._run_one_simulation_step()
        bridge.cp_external_input_current[:] = 0.0

        self.bridge = bridge
        self.xp = xp
        self.meta_idx = xp.asarray(meta)
        self.self_idx = xp.asarray(self_idx)
        self._snap = _snapshot_state(bridge, xp)

    def _run_rate(self, confidence: float | None) -> float:
        if confidence is None:
            confidence = 0.0
        c = float(np.clip(confidence, 0.0, 1.0))
        current = c * float(self.config.max_meta_current_pa)
        bridge = self.bridge
        xp = self.xp
        _restore_state(bridge, self._snap)
        bridge.cp_external_input_current[:] = 0.0
        steps = max(3, int(self.config.report_steps))
        late_start = steps - max(1, steps // 2)
        acc = 0
        for t in range(steps):
            bridge.cp_external_input_current[:] = 0.0
            bridge.cp_external_input_current[self.meta_idx] = xp.float32(current)
            bridge._run_one_simulation_step()
            if t >= late_start:
                acc += int(to_host(bridge.cp_firing_states[self.self_idx].astype(xp.float64).sum()))
        bridge.cp_external_input_current[:] = 0.0
        return acc / (float(steps - late_start) * float(self.config.n_self))

    def read(self, source_confidence: float | None, *, familiar: bool) -> dict[str, Any]:
        """Return a self-schema confidence record for a matched answer."""
        self_rate = self._run_rate(source_confidence)
        if source_confidence is None:
            band = "unproven"
        elif self.config.require_source_floor and source_confidence < self.config.confidence_hedge:
            band = "soft_abstain"
        elif self.config.require_source_floor and source_confidence < self.config.confidence_assert:
            band = "hedge" if self_rate >= self._hedge_rate else "soft_abstain"
        elif self_rate >= self._assert_rate:
            band = "assert"
        elif self_rate >= self._hedge_rate:
            band = "hedge"
        else:
            band = "soft_abstain"
        return {
            "source": "composer_trace_to_self_schema",
            "familiar": bool(familiar),
            "source_confidence": None if source_confidence is None else float(source_confidence),
            "self_schema_rate": float(self_rate),
            "assert_rate_threshold": float(self._assert_rate),
            "hedge_rate_threshold": float(self._hedge_rate),
            "band": band,
            "certain": bool(band == "assert"),
            "config": asdict(self.config),
            "lesion_self_read": bool(self.lesion_self_read),
        }
