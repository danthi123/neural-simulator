"""Bounded v2 calibration for replay-driven cortical consolidation.

V1 established a causal CA3 -> CA1 -> cortex path, but its cortical replay was
seed-fragile: one calibration brain learned broad false associations while the
other barely learned. V2 changes the mechanism, not the host stimulus strength.
Each cortical target assembly now excites a local fast-spiking interneuron
channel that inhibits the competing target assembly. Assembly recurrence can
therefore amplify a coherent reinstatement while opponent inhibition suppresses
diffuse competitors without globally silencing the desired assembly.

Sleep input remains episode-agnostic. The host creates a deterministic random
walk of weak CA3 background subsets and never chooses A or B. The
``shuffled_replay_order`` control permutes that exact event list while
preserving every stimulated cell in every event. Slow Hebbian coactivity state
persists across events, so order can affect consolidation; membrane and fast
conductance state are reset between events to model down-states.

Remaining scaffolds are explicit:

* wake episode populations and partial probe cues are host-defined;
* opponent inhibitory channel membership follows the fixed host-defined
  calibration assemblies rather than developing from experience;
* sleep down-state boundaries and episode-agnostic CA3 background current are
  scheduled by the host;
* host code reads spikes, weights, and known calibration assemblies for
  measurement only;
* the rate-window Hebbian rule and fixed assembly anatomy are simplified
  biological stand-ins, not a complete sleep circuit.

Only calibration seeds 212 and 213 are accepted. Development seeds 214, 215,
310 and held-out seeds 311, 312, 313 are mechanically rejected.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from research.runners import _replay_cortical_consolidation_gate as v1  # noqa: E402
from tools.lab import attributable_to  # noqa: E402
from tools.verdict import UNDEFINED, Verdict  # noqa: E402


CALIBRATION_SEEDS = (212, 213)
DEVELOPMENT_SEEDS = (214, 215, 310)
HELD_OUT_SEEDS = (311, 312, 313)
CONDITIONS = (
    "intact",
    "no_sleep",
    "shuffled_replay_order",
    "shuffled_target_index",
    "ca3_ca1_lesion",
    "cortical_plasticity_off",
    "target_inhibition_lesion",
)

CA3_GATE = v1.CA3_GATE
INDEX_CUE_GATE = v1.INDEX_CUE_GATE
INDEX_TARGET_GATE = v1.INDEX_TARGET_GATE
CORTICAL_GATE = v1.CORTICAL_GATE
SCHAFFER_GATE = v1.SCHAFFER_GATE
TARGET_INHIBITION_GATE = "replay_v2_target_fs_to_pyramidal"
TARGET_RECURRENT_GATE = "replay_v2_target_recurrent_fixed"


@dataclass(frozen=True)
class GateConfig(v1.GateConfig):
    """V2 anatomy and timing; external drive magnitudes stay at v1 values."""

    n_target_fs: int = 12
    ca1_to_fs_weight: float = 40.0
    target_to_fs_weight: float = 120.0
    fs_to_target_weight: float = 44.0
    cortical_target_recurrent_weight: float = 24.0
    sleep_turnover_cells: int = 3
    sleep_free_steps: int = 20


def smoke_config() -> GateConfig:
    return GateConfig(
        n_ca3=36,
        n_ca1=24,
        n_cue=24,
        n_target=24,
        n_target_fs=6,
        ca3_assembly=12,
        ca3_overlap=3,
        ca1_assembly=8,
        cue_assembly=8,
        cue_overlap=3,
        target_assembly=8,
        encode_a_events=3,
        encode_b_events=4,
        encode_on_steps=6,
        encode_off_steps=4,
        sleep_events=6,
        sleep_noise_cells=5,
        sleep_noise_steps=3,
        sleep_free_steps=20,
        sleep_turnover_cells=1,
        probe_steps=12,
    )


def validate_calibration_seeds(seeds: Iterable[int]) -> tuple[int, ...]:
    checked = tuple(int(seed) for seed in seeds)
    invalid = [seed for seed in checked if seed not in CALIBRATION_SEEDS]
    if invalid:
        raise ValueError(
            f"This bounded v2 runner accepts calibration seeds {CALIBRATION_SEEDS} only; "
            f"refusing reserved seeds {invalid}."
        )
    if not checked:
        raise ValueError("At least one calibration seed is required.")
    return checked


def _population(
    pre: np.ndarray,
    post: np.ndarray,
    weight: float,
    *,
    plastic: bool,
    plasticity_gate: str | None = None,
    transmission_gate: str | None = None,
    conn_type: str = "E_TO_E",
    self_edges: bool = True,
) -> dict:
    edge_pre, edge_post = v1._all_to_all(pre, post, self_edges=self_edges)
    row = {
        "pre_indices": edge_pre.tolist(),
        "post_indices": edge_post.tolist(),
        "initial_weights": np.full(edge_pre.size, weight, dtype=np.float32),
        "plastic": bool(plastic),
        "conn_type": conn_type,
        "count": int(edge_pre.size),
    }
    if plasticity_gate is not None:
        row["plasticity_gate"] = plasticity_gate
    if transmission_gate is not None:
        row["transmission_gate"] = transmission_gate
    return row


def _merge_pairs(
    pairs: list[tuple[np.ndarray, np.ndarray]],
    weight: float,
    **kwargs,
) -> dict:
    rows = [_population(pre, post, weight, **kwargs) for pre, post in pairs]
    merged = {
        "pre_indices": sum((row["pre_indices"] for row in rows), []),
        "post_indices": sum((row["post_indices"] for row in rows), []),
        "initial_weights": np.concatenate([row["initial_weights"] for row in rows]),
        "plastic": rows[0]["plastic"],
        "conn_type": rows[0]["conn_type"],
        "count": sum(row["count"] for row in rows),
    }
    for key in ("plasticity_gate", "transmission_gate"):
        if key in rows[0]:
            merged[key] = rows[0][key]
    return merged


def build_bridge(seed: int, config: GateConfig) -> tuple[object, dict]:
    """Build one bridge with a local cortical pyramidal/FS competition loop."""
    from sim.backend import get_backend
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, GPUConfig, RuntimeState, VisualizationConfig
    from sim.enums import NeuronModel, NeuronType
    from sim.regions import BrainRegion, RegionPathway

    excitatory = dict(
        exc_fraction=1.0,
        internal_density=0.0,
        exc_weight_mean=0.0,
        inh_weight_mean=0.0,
        weight_jitter=0.0,
        plastic_internal=False,
        izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name,
    )
    inhibitory = dict(
        exc_fraction=0.0,
        internal_density=0.0,
        exc_weight_mean=0.0,
        inh_weight_mean=0.0,
        weight_jitter=0.0,
        plastic_internal=False,
        izh_neuron_type=NeuronType.IZH2007_FS_CORTICAL_INTERNEURON.name,
    )
    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = [
        BrainRegion("ca3", config.n_ca3, **excitatory),
        BrainRegion("ca1", config.n_ca1, **excitatory),
        BrainRegion("cortical_cue", config.n_cue, **excitatory),
        BrainRegion("cortical_target", config.n_target, **excitatory),
        BrainRegion("cortical_target_fs", config.n_target_fs, **inhibitory),
    ]
    cfg.region_pathways = [
        RegionPathway("ca3", "ca1", density=0.01, weight_mean=0.01, plastic=False),
    ]
    cfg.num_neurons = 0
    cfg.connections_per_neuron = 0
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.dt_ms = 1.0
    cfg.seed = cfg.heterogeneity_seed = cfg.ou_seed = int(seed)
    cfg.enable_stdp = False
    cfg.enable_hebbian_learning = True
    cfg.hebbian_rate_window = True
    cfg.hebbian_learning_rate = float(config.hebbian_learning_rate)
    cfg.hebbian_max_weight = float(config.hebbian_max_weight)
    cfg.hebbian_min_weight = 0.0
    cfg.hebbian_weight_decay = 0.0
    cfg.hebbian_coactivity_decay = float(config.hebbian_coactivity_decay)
    cfg.hebbian_coactivity_thresh = float(config.hebbian_coactivity_thresh)
    cfg.enable_reward_modulation = False
    cfg.enable_homeostasis = False
    cfg.enable_short_term_plasticity = False
    cfg.enable_structural_plasticity = False
    cfg.enable_ou_process = False
    cfg.ou_std_current_pA = 0.0
    cfg.fast_spike_reset = True
    cfg.propagation_strength = float(config.propagation_strength)

    runtime = RuntimeState()
    runtime.actual_seed_used = int(seed)
    bridge = SimulationBridge(
        core_config=cfg,
        viz_config=VisualizationConfig(),
        runtime_state=runtime,
        gpu_config=GPUConfig(),
    )
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)

    region_names = ("ca3", "ca1", "cortical_cue", "cortical_target", "cortical_target_fs")
    regions = {
        name: np.asarray(bridge.region_manager.indices(name), dtype=np.int64)
        for name in region_names
    }
    patterns = v1._memory_patterns(seed, config, regions)
    if config.n_target_fs % 2:
        raise ValueError("n_target_fs must be even so A/B opponent channels are balanced")
    fs_a, fs_b = np.split(regions["cortical_target_fs"], 2)

    ca3_edges = []
    for memory in ("A", "B"):
        edge_pre, edge_post = v1._all_to_all(
            patterns[memory]["ca3"], patterns[memory]["ca3"], self_edges=False,
        )
        ca3_edges.append(np.column_stack([edge_pre, edge_post]))
    ca3_edges = np.unique(np.concatenate(ca3_edges, axis=0), axis=0)
    ca3_recurrent = {
        "pre_indices": ca3_edges[:, 0].astype(np.int64).tolist(),
        "post_indices": ca3_edges[:, 1].astype(np.int64).tolist(),
        "initial_weights": np.full(ca3_edges.shape[0], config.ca3_initial_weight, dtype=np.float32),
        "plastic": True,
        "plasticity_gate": CA3_GATE,
        "conn_type": "E_TO_E",
        "count": int(ca3_edges.shape[0]),
    }
    ca3_to_ca1 = []
    index_cue = []
    index_target = []
    recurrent_target = []
    for memory in ("A", "B"):
        pat = patterns[memory]
        ca3_to_ca1.append((pat["ca3"], pat["ca1"]))
        index_cue.append((pat["ca1"], pat["cue"]))
        index_target.append((pat["ca1"], pat["target"]))
        recurrent_target.append((pat["target"], pat["target"]))

    wiring = {
        "ca3_recurrent": ca3_recurrent,
        "ca3_to_ca1": _merge_pairs(
            ca3_to_ca1,
            config.ca3_to_ca1_weight,
            plastic=False,
            plasticity_gate=SCHAFFER_GATE,
            transmission_gate=SCHAFFER_GATE,
        ),
        "ca1_to_cortical_cue": _merge_pairs(
            index_cue,
            config.index_initial_weight,
            plastic=True,
            plasticity_gate=INDEX_CUE_GATE,
        ),
        "ca1_to_cortical_target": _merge_pairs(
            index_target,
            config.index_initial_weight,
            plastic=True,
            plasticity_gate=INDEX_TARGET_GATE,
        ),
        "cortical_association": _population(
            regions["cortical_cue"],
            regions["cortical_target"],
            config.cortical_initial_weight,
            plastic=True,
            plasticity_gate=CORTICAL_GATE,
        ),
        "cortical_target_recurrent": _merge_pairs(
            recurrent_target,
            config.cortical_target_recurrent_weight,
            plastic=False,
            plasticity_gate=TARGET_RECURRENT_GATE,
            self_edges=False,
        ),
        "target_to_fs": _merge_pairs(
            [
                (patterns["A"]["target"], fs_a),
                (patterns["B"]["target"], fs_b),
            ],
            config.target_to_fs_weight,
            plastic=False,
            conn_type="E_TO_I",
        ),
        "ca1_to_target_fs": _merge_pairs(
            [
                (patterns["A"]["ca1"], fs_a),
                (patterns["B"]["ca1"], fs_b),
            ],
            config.ca1_to_fs_weight,
            plastic=False,
            conn_type="E_TO_I",
        ),
        "fs_to_target": _merge_pairs(
            [
                (fs_a, patterns["B"]["target"]),
                (fs_b, patterns["A"]["target"]),
            ],
            config.fs_to_target_weight,
            plastic=False,
            transmission_gate=TARGET_INHIBITION_GATE,
            conn_type="I_TO_E",
        ),
    }
    bridge.inject_explicit_wiring(
        wiring,
        output_inhibitory_indices=regions["cortical_target_fs"].tolist(),
    )
    bridge.set_plasticity_gate(TARGET_RECURRENT_GATE, 0.0)
    bridge.set_transmission_gate(TARGET_INHIBITION_GATE, 1.0)

    xp, _ = get_backend()
    handles = {
        "regions": regions,
        "patterns": patterns,
        "device_patterns": {
            memory: {key: xp.asarray(value, dtype=xp.int64) for key, value in pat.items()}
            for memory, pat in patterns.items()
        },
        "wiring_counts": {name: int(row["count"]) for name, row in wiring.items()},
        "bridge_identity": id(bridge),
        "inhibitory_indices": regions["cortical_target_fs"].copy(),
        "fs_pools": {"A": fs_a.copy(), "B": fs_b.copy()},
    }
    return bridge, handles


def _sleep_event_plan(seed: int, config: GateConfig, ca3_indices: np.ndarray) -> list[np.ndarray]:
    """Episode-agnostic correlated CA3 background events, independent of A/B labels."""
    rng = np.random.default_rng(seed * 59 + 11)
    current = np.sort(rng.choice(ca3_indices, config.sleep_noise_cells, replace=False))
    events = [current.copy()]
    turnover = min(config.sleep_turnover_cells, config.sleep_noise_cells)
    for _ in range(1, config.sleep_events):
        keep_n = config.sleep_noise_cells - turnover
        kept = rng.choice(current, keep_n, replace=False) if keep_n else np.empty(0, dtype=np.int64)
        available = np.setdiff1d(ca3_indices, kept, assume_unique=False)
        added = rng.choice(available, turnover, replace=False)
        current = np.sort(np.concatenate([kept, added])).astype(np.int64)
        events.append(current.copy())
    return events


def _ordered_sleep_events(
    seed: int,
    config: GateConfig,
    ca3_indices: np.ndarray,
    *,
    shuffle: bool,
) -> list[np.ndarray]:
    events = _sleep_event_plan(seed, config, ca3_indices)
    if not shuffle or len(events) < 2:
        return events
    order = np.random.default_rng(seed * 71 + 19).permutation(len(events))
    if np.array_equal(order, np.arange(len(events))):
        order = np.roll(order, 1)
    return [events[int(index)].copy() for index in order]


def _event_digest(events: list[np.ndarray], *, order_sensitive: bool) -> str:
    rows = [",".join(str(int(cell)) for cell in event) for event in events]
    if not order_sensitive:
        rows.sort()
    return hashlib.sha256("|".join(rows).encode("ascii")).hexdigest()


def _mean_adjacent_overlap(events: list[np.ndarray]) -> float:
    if len(events) < 2:
        return 0.0
    overlaps = []
    for left, right in zip(events, events[1:]):
        union = np.union1d(left, right)
        overlaps.append(len(np.intersect1d(left, right)) / max(len(union), 1))
    return float(np.mean(overlaps))


def _clear_fast_dynamics(bridge) -> None:
    """Reset a down-state while preserving learned weights and slow Hebbian trace."""
    if getattr(bridge, "cp_izh_c_reset", None) is not None:
        bridge.cp_membrane_potential_v[:] = bridge.cp_izh_c_reset
    else:
        bridge.cp_membrane_potential_v[:] = -65.0
    if getattr(bridge, "cp_recovery_variable_u", None) is not None:
        bridge.cp_recovery_variable_u[:] = 0.0
    for name in (
        "cp_firing_states",
        "cp_prev_firing_states",
        "cp_conductance_g_e",
        "cp_conductance_g_i",
        "cp_conductance_g_nmda",
        "cp_conductance_g_nmda_rise",
        "cp_conductance_g_nmda_recurrent",
        "cp_conductance_g_nmda_recurrent_rise",
    ):
        array = getattr(bridge, name, None)
        if array is not None:
            array[:] = 0
    v1._zero_current(bridge)


def _sleep(bridge, handles: dict, condition: str, seed: int, config: GateConfig) -> dict:
    from sim.backend import get_backend

    xp, _ = get_backend()
    v1._clear_dynamics(bridge)
    cortical_on = condition != "cortical_plasticity_off"
    v1._set_phase_gates(bridge, sleep=True, cortical=cortical_on)
    bridge.core_config.hebbian_learning_rate = float(config.cortical_sleep_learning_rate)
    if condition == "ca3_ca1_lesion":
        bridge.set_transmission_gate(SCHAFFER_GATE, 0.0)
    inhibition_gain = 0.0 if condition == "target_inhibition_lesion" else 1.0
    bridge.set_transmission_gate(TARGET_INHIBITION_GATE, inhibition_gain)
    shuffled_edges = (
        v1._shuffle_target_index(bridge, seed)
        if condition == "shuffled_target_index"
        else 0
    )

    regions = handles["regions"]
    events = _ordered_sleep_events(
        seed,
        config,
        regions["ca3"],
        shuffle=condition == "shuffled_replay_order",
    )
    ca3_dev = xp.asarray(regions["ca3"], dtype=xp.int64)
    ca1_dev = xp.asarray(regions["ca1"], dtype=xp.int64)
    cue_dev = xp.asarray(regions["cortical_cue"], dtype=xp.int64)
    target_dev = xp.asarray(regions["cortical_target"], dtype=xp.int64)
    fs_dev = xp.asarray(regions["cortical_target_fs"], dtype=xp.int64)
    ca3_a = handles["device_patterns"]["A"]["ca3"]
    ca3_b = handles["device_patterns"]["B"]["ca3"]
    event_winners: list[str] = []
    spike_totals = {
        "ca3": 0,
        "ca1": 0,
        "cortical_cue": 0,
        "cortical_target": 0,
        "cortical_target_fs": 0,
    }

    for event in events:
        _clear_fast_dynamics(bridge)
        event_a = event_b = 0
        if condition == "no_sleep":
            v1._step(bridge, config.sleep_noise_steps + config.sleep_free_steps)
        else:
            background_dev = xp.asarray(event, dtype=xp.int64)
            for step in range(config.sleep_noise_steps + config.sleep_free_steps):
                v1._zero_current(bridge)
                if step < config.sleep_noise_steps:
                    bridge.cp_external_input_current[background_dev] = config.sleep_drive_pA
                bridge._run_one_simulation_step()
                bridge.runtime_state.current_time_step += 1
                firing = bridge.cp_firing_states
                event_a += int(firing[ca3_a].sum())
                event_b += int(firing[ca3_b].sum())
                spike_totals["ca3"] += int(firing[ca3_dev].sum())
                spike_totals["ca1"] += int(firing[ca1_dev].sum())
                spike_totals["cortical_cue"] += int(firing[cue_dev].sum())
                spike_totals["cortical_target"] += int(firing[target_dev].sum())
                spike_totals["cortical_target_fs"] += int(firing[fs_dev].sum())
        if event_a == event_b == 0:
            event_winners.append("none")
        elif event_a > event_b:
            event_winners.append("A")
        elif event_b > event_a:
            event_winners.append("B")
        else:
            event_winners.append("tie")

    v1._zero_current(bridge)
    bridge.set_transmission_gate(TARGET_INHIBITION_GATE, 1.0)
    return {
        "events": int(len(events)),
        "event_winners": event_winners,
        "reactivated_events": int(sum(winner != "none" for winner in event_winners)),
        "replayed_A": int(sum(winner == "A" for winner in event_winners)),
        "replayed_B": int(sum(winner == "B" for winner in event_winners)),
        "spikes": spike_totals,
        "shuffled_edges": int(shuffled_edges),
        "event_content_multiset_digest": _event_digest(events, order_sensitive=False),
        "event_order_digest": _event_digest(events, order_sensitive=True),
        "mean_adjacent_input_overlap": _mean_adjacent_overlap(events),
        "target_inhibition_gain_during_sleep": inhibition_gain,
        "host_selected_episode_for_replay": False,
    }


def run_condition(seed: int, condition: str, config: GateConfig | None = None) -> dict:
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition {condition!r}; expected one of {CONDITIONS}.")
    validate_calibration_seeds([seed])
    cfg = config or GateConfig()
    bridge, handles = build_bridge(seed, cfg)
    bridge_ids = [id(bridge)]
    phase_trace: list[str] = []

    # The candidate circuit is a sleep/retrieval competition mechanism. Wake
    # sensory teaching must first establish the hippocampal index without that
    # same loop suppressing the externally presented target population.
    bridge.set_transmission_gate(TARGET_INHIBITION_GATE, 0.0)

    before = {
        "ca3": v1._path_weights(bridge, CA3_GATE),
        "index_cue": v1._path_weights(bridge, INDEX_CUE_GATE),
        "index_target": v1._path_weights(bridge, INDEX_TARGET_GATE),
        "cortical": v1._path_weights(bridge, CORTICAL_GATE),
    }
    encode_a = v1._encode_memory(bridge, handles, "A", cfg.encode_a_events, cfg)
    phase_trace.append("encode_A")
    bridge_ids.append(id(bridge))
    after_a = {
        "ca3": v1._path_weights(bridge, CA3_GATE),
        "cortical": v1._path_weights(bridge, CORTICAL_GATE),
    }
    encode_b = v1._encode_memory(bridge, handles, "B", cfg.encode_b_events, cfg)
    phase_trace.append("encode_B")
    bridge_ids.append(id(bridge))
    after_b = {
        "ca3": v1._path_weights(bridge, CA3_GATE),
        "index_cue": v1._path_weights(bridge, INDEX_CUE_GATE),
        "index_target": v1._path_weights(bridge, INDEX_TARGET_GATE),
        "cortical": v1._path_weights(bridge, CORTICAL_GATE),
    }
    sleep = _sleep(bridge, handles, condition, seed, cfg)
    phase_trace.append("sleep")
    bridge_ids.append(id(bridge))
    after_sleep = {
        "ca3": v1._path_weights(bridge, CA3_GATE),
        "index_cue": v1._path_weights(bridge, INDEX_CUE_GATE),
        "index_target": v1._path_weights(bridge, INDEX_TARGET_GATE),
        "cortical": v1._path_weights(bridge, CORTICAL_GATE),
    }
    recall = {memory: v1._probe_memory(bridge, handles, memory, cfg) for memory in ("A", "B")}
    phase_trace.append("retest")
    bridge_ids.append(id(bridge))

    def mean_delta(later: np.ndarray, earlier: np.ndarray) -> float:
        return float(np.mean(later - earlier))

    return {
        "seed": int(seed),
        "condition": condition,
        "config": asdict(cfg),
        "phase_trace": phase_trace,
        "single_bridge_persisted": len(set(bridge_ids)) == 1,
        "wiring_counts": handles["wiring_counts"],
        "inhibitory_neuron_count": int(len(handles["inhibitory_indices"])),
        "encode_A": encode_a,
        "encode_B": encode_b,
        "sleep": sleep,
        "recall": recall,
        "weight_deltas": {
            "ca3_during_encode_A": mean_delta(after_a["ca3"], before["ca3"]),
            "ca3_during_encode_B": mean_delta(after_b["ca3"], after_a["ca3"]),
            "ca3_during_sleep": mean_delta(after_sleep["ca3"], after_b["ca3"]),
            "index_cue_during_wake": mean_delta(after_b["index_cue"], before["index_cue"]),
            "index_target_during_wake": mean_delta(after_b["index_target"], before["index_target"]),
            "index_cue_during_sleep": mean_delta(after_sleep["index_cue"], after_b["index_cue"]),
            "index_target_during_sleep": mean_delta(after_sleep["index_target"], after_b["index_target"]),
            "cortical_during_wake": mean_delta(after_b["cortical"], before["cortical"]),
            "cortical_during_sleep": mean_delta(after_sleep["cortical"], after_b["cortical"]),
        },
    }


def _mean_recovery(row: dict) -> float:
    return float(np.mean([row["recall"][memory]["correct_rate"] for memory in ("A", "B")]))


def _mean_margin(row: dict) -> float:
    return float(np.mean([row["recall"][memory]["margin"] for memory in ("A", "B")]))


def _mean_false_recall(row: dict) -> float:
    return float(np.mean([row["recall"][memory]["false_recall_fraction"] for memory in ("A", "B")]))


def _calibration_verdict(conditions: dict[str, dict]) -> dict:
    intact = conditions["intact"]
    controls = {name: row for name, row in conditions.items() if name != "intact"}
    intact_recovery = _mean_recovery(intact)
    intact_margin = _mean_margin(intact)
    intact_false = _mean_false_recall(intact)
    control_recovery = {name: _mean_recovery(row) for name, row in controls.items()}
    attribution = {
        name: attributable_to(f"v2 replay consolidation versus {name}", intact_recovery, recovery)
        for name, recovery in control_recovery.items()
    }
    temporal = conditions["shuffled_replay_order"]
    plastic_off = conditions["cortical_plasticity_off"]
    inhibition_lesion = conditions["target_inhibition_lesion"]
    checks = {
        "single_bridge_all_conditions": all(row["single_bridge_persisted"] for row in conditions.values()),
        "both_memories_replayed": intact["sleep"]["replayed_A"] > 0 and intact["sleep"]["replayed_B"] > 0,
        "local_inhibition_recruited": intact["sleep"]["spikes"]["cortical_target_fs"] > 0,
        "temporal_control_preserves_event_content": (
            intact["sleep"]["event_content_multiset_digest"]
            == temporal["sleep"]["event_content_multiset_digest"]
        ),
        "temporal_control_changes_order": (
            intact["sleep"]["event_order_digest"] != temporal["sleep"]["event_order_digest"]
            and intact["sleep"]["mean_adjacent_input_overlap"]
            > temporal["sleep"]["mean_adjacent_input_overlap"]
        ),
        "cortical_weights_change_only_during_sleep": (
            intact["weight_deltas"]["cortical_during_sleep"] > 1e-5
            and abs(intact["weight_deltas"]["cortical_during_wake"]) < 1e-7
            and abs(plastic_off["weight_deltas"]["cortical_during_sleep"]) < 1e-7
        ),
        "intact_partial_recovery": intact_recovery >= 0.03 and intact_margin >= 0.015,
        "both_memories_recovered": all(
            intact["recall"][memory]["correct_rate"] >= 0.01 for memory in ("A", "B")
        ),
        "false_recall_bounded": intact_false <= 0.15,
        "intact_beats_no_sleep": intact_recovery >= control_recovery["no_sleep"] + 0.015,
        "intact_beats_shuffled_order": intact_recovery >= control_recovery["shuffled_replay_order"] + 0.01,
        "learned_target_index_beats_shuffle": (
            intact_recovery >= control_recovery["shuffled_target_index"] + 0.015
        ),
        "schaffer_path_is_load_bearing": (
            intact_recovery >= control_recovery["ca3_ca1_lesion"] + 0.015
        ),
        "cortical_plasticity_is_load_bearing": (
            intact_recovery >= control_recovery["cortical_plasticity_off"] + 0.015
        ),
        "target_inhibition_improves_specificity": (
            intact_false <= _mean_false_recall(inhibition_lesion) - 0.05
            and intact_recovery >= 0.75 * control_recovery["target_inhibition_lesion"]
        ),
    }

    earned = Verdict("replay-driven cortical consolidation v2 calibration")
    earned.require(
        "one bridge persists through every phase and condition",
        checks["single_bridge_all_conditions"],
        expect=True,
    )
    earned.require(
        "all conditions execute the fixed phase sequence",
        all(row["phase_trace"] == ["encode_A", "encode_B", "sleep", "retest"] for row in conditions.values()),
        expect=True,
    )
    earned.require(
        "both wake episodes recruit every required excitatory region",
        all(
            row[f"encode_{memory}"]["spikes"][region] > 0
            for row in conditions.values()
            for memory in ("A", "B")
            for region in ("ca3", "ca1", "cue", "target")
        ),
        expect=True,
    )
    earned.require(
        "intact sleep contains uncued replay and FS inhibitory spikes",
        intact["sleep"]["reactivated_events"] > 0 and checks["local_inhibition_recruited"],
        expect=True,
    )
    earned.require(
        "temporal control changes order while preserving exact event content",
        checks["temporal_control_preserves_event_content"] and checks["temporal_control_changes_order"],
        expect=True,
    )
    earned.require(
        "no-sleep control remains quiescent",
        conditions["no_sleep"]["sleep"]["reactivated_events"] == 0
        and sum(conditions["no_sleep"]["sleep"]["spikes"].values()) == 0,
        expect=True,
    )
    earned.require(
        "inhibition lesion reaches the FS output transmission gate",
        inhibition_lesion["sleep"]["target_inhibition_gain_during_sleep"] == 0.0
        and intact["sleep"]["target_inhibition_gain_during_sleep"] == 1.0,
        expect=True,
    )
    earned.require(
        "cortical-plasticity-off holds cortical weights fixed",
        abs(plastic_off["weight_deltas"]["cortical_during_sleep"]) < 1e-7,
        expect=True,
    )
    earned.disabled(
        "STDP, reward modulation, homeostasis, short-term plasticity, and structural plasticity",
        why="bounded isolation of Hebbian replay transfer on fixed anatomy",
    )
    decided = earned.decide(go=all(checks.values()), verbose=False)
    return {
        "calibration_status": (
            "UNDEFINED"
            if decided["status"] == UNDEFINED
            else "CALIBRATION_PROMISING" if decided["go"] else "CALIBRATION_NEEDS_REVISION"
        ),
        "preconditions": decided["preconditions"],
        "disabled_processes": decided["disabled_processes"],
        "undefined_reasons": decided["undefined_reasons"],
        "checks": checks,
        "intact_mean_recovery": intact_recovery,
        "intact_mean_margin": intact_margin,
        "intact_mean_false_recall": intact_false,
        "control_mean_recovery": control_recovery,
        "control_mean_false_recall": {
            name: _mean_false_recall(row) for name, row in controls.items()
        },
        "attribution": attribution,
    }


def run_seed(seed: int, config: GateConfig | None = None) -> dict:
    validate_calibration_seeds([seed])
    cfg = config or GateConfig()
    conditions = {condition: run_condition(seed, condition, cfg) for condition in CONDITIONS}
    verdict = _calibration_verdict(conditions)
    return {
        "seed": int(seed),
        "conditions": conditions,
        "calibration": verdict,
        "calibration_status": verdict["calibration_status"],
    }


def run_calibration(seeds: Iterable[int], config: GateConfig | None = None) -> dict:
    checked = validate_calibration_seeds(seeds)
    started = time.time()
    rows = [run_seed(seed, config) for seed in checked]
    statuses = [row["calibration_status"] for row in rows]
    if any(status == "UNDEFINED" for status in statuses):
        aggregate_status = "UNDEFINED"
    elif all(status == "CALIBRATION_PROMISING" for status in statuses):
        aggregate_status = "CALIBRATION_PROMISING"
    else:
        aggregate_status = "CALIBRATION_NEEDS_REVISION"
    return {
        "gate": "replay_cortical_consolidation_v2",
        "phase": "calibration",
        "calibration_status": aggregate_status,
        "seeds": list(checked),
        "reserved_seeds_inspected": False,
        "rows": rows,
        "remaining_scaffolds": [
            "host-defined wake episode populations and partial probe cues",
            "opponent inhibitory channel membership fixed from calibration assemblies",
            "host-scheduled sleep down-state boundaries and episode-agnostic CA3 background current",
            "host spike/weight measurement against known calibration assemblies",
            "rate-window Hebbian plasticity and fixed assembly anatomy",
        ],
        "elapsed_seconds": time.time() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(CALIBRATION_SEEDS))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    payload = run_calibration(args.seeds, smoke_config() if args.smoke else GateConfig())
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
