"""Bounded v3 calibration for replay-driven cortical consolidation.

V2 made cortical replay selective with opponent fast-spiking inhibition, but
the learned CA1 target index was not reliably stronger than a shuffled index
and one memory could remain much weaker than the other. V3 adds two local
mechanisms while retaining v2's replay and causal controls:

* CA1 projects broadly to a cortical index-relay population. Wake coactivity
  learns the CA1-to-relay mapping; no target-specific host current is delivered
  to the relay. A weight-sensitive dendritic coincidence response and local
  recurrence reinstate the learned relay during sleep, which then excites the
  associated cortical target assembly.
* Each relay assembly recruits its own slow GABA-B feedback interneurons.
  Strongly replayed relays therefore adapt locally across successive replay
  events, reducing monopoly by one memory without host ranking or selection.

The sleep event generator and shuffled-order control are inherited unchanged
from v2: the control permutes the exact event list, preserving every stimulated
cell and the event-content multiset. Opponent target inhibition also remains.

Remaining scaffolds are explicit: wake assemblies and partial probes are
host-defined; relay, teacher, and inhibitory channel membership use fixed
calibration anatomy; sleep down-states and episode-agnostic CA3 background
current are host-scheduled; host code reads activity and known assemblies only
for measurement; and the fixed anatomy and rate-window Hebbian rule remain
simplified biological stand-ins.

The scientific verdict path is retired. Seed 216 remains reserved solely for
non-scientific smoke tests and is outside every scientific partition.
Calibration seeds 228/229 were consumed by an undefined calibration and are
closed; development seeds 230/231/326 and held-out seeds 327/328/329 remain
mechanically rejected. No scientific phase is open.
"""
from __future__ import annotations

import argparse
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
from research.runners import _replay_cortical_consolidation_gate_v2 as v2  # noqa: E402
from tools.lab import attributable_to  # noqa: E402
from tools.verdict import UNDEFINED, Verdict  # noqa: E402


OPEN_PHASES: tuple[str, ...] = ()
SMOKE_SEED = 216
CALIBRATION_SEEDS = (228, 229)
DEVELOPMENT_SEEDS = (230, 231, 326)
HELD_OUT_SEEDS = (327, 328, 329)
CONDITIONS = (
    "intact",
    "no_sleep",
    "shuffled_replay_order",
    "shuffled_target_index",
    "ca3_ca1_lesion",
    "cortical_plasticity_off",
    "target_inhibition_lesion",
    "index_relay_lesion",
    "index_balance_lesion",
)

CA3_GATE = v1.CA3_GATE
INDEX_CUE_GATE = v1.INDEX_CUE_GATE
INDEX_TARGET_GATE = v1.INDEX_TARGET_GATE
CORTICAL_GATE = v1.CORTICAL_GATE
SCHAFFER_GATE = v1.SCHAFFER_GATE
TARGET_INHIBITION_GATE = v2.TARGET_INHIBITION_GATE
TARGET_RECURRENT_GATE = v2.TARGET_RECURRENT_GATE
INDEX_RECURRENT_GATE = "replay_v3_index_recurrent_fixed"
INDEX_OUTPUT_GATE = "replay_v3_index_to_target"
INDEX_BALANCE_GATE = "replay_v3_index_gabab_feedback"
WAKE_TEACHING_GATE = "replay_v3_target_to_index_wake"


@dataclass(frozen=True)
class GateConfig(v2.GateConfig):
    """V3 local relay/adaptation anatomy; global drive and learning rates stay fixed."""

    n_index: int = 32
    n_index_fs: int = 8
    index_assembly: int = 12
    target_to_index_weight: float = 72.0
    index_to_target_weight: float = 34.0
    index_recurrent_weight: float = 20.0
    index_to_fs_weight: float = 100.0
    fs_to_index_weight: float = 24.0
    index_coincidence_threshold: float = 90.0
    index_plateau_strength: float = 70.0
    index_gabab_tau_ms: float = 120.0
    index_gabab_propagation: float = 0.055


def smoke_config() -> GateConfig:
    return GateConfig(
        n_ca3=36,
        n_ca1=24,
        n_cue=24,
        n_target=24,
        n_target_fs=6,
        n_index=20,
        n_index_fs=4,
        ca3_assembly=12,
        ca3_overlap=3,
        ca1_assembly=8,
        cue_assembly=8,
        cue_overlap=3,
        target_assembly=8,
        index_assembly=8,
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
        index_coincidence_threshold=35.0,
    )


def validate_phase(phase: str) -> str:
    checked = str(phase).strip().lower()
    if checked not in OPEN_PHASES:
        raise ValueError(
            "This bounded v3 runner has no open scientific phase; calibration "
            f"seeds {CALIBRATION_SEEDS} are consumed and closed. Refusing {phase!r}."
        )
    return checked


def validate_calibration_seed(seed: int) -> int:
    checked = int(seed)
    if checked in CALIBRATION_SEEDS:
        raise ValueError(
            f"Replay-v3 calibration seed {checked} is consumed and closed."
        )
    else:
        raise ValueError(
            "Replay-v3 has no open calibration partition; refusing reserved "
            f"seed {checked}."
        )


def validate_calibration_seeds(seeds: Iterable[int]) -> tuple[int, ...]:
    checked = tuple(int(seed) for seed in seeds)
    if checked != CALIBRATION_SEEDS:
        raise ValueError(
            f"Calibration historically used the exact ordered calibration seed partition "
            f"{CALIBRATION_SEEDS}; refusing {checked}."
        )
    raise ValueError(
        f"Replay-v3 calibration seeds {CALIBRATION_SEEDS} are consumed and closed."
    )


def validate_smoke_seed(seed: int) -> int:
    checked = int(seed)
    if checked != SMOKE_SEED:
        raise ValueError(
            f"Smoke execution accepts non-scientific seed {SMOKE_SEED} only; "
            f"refusing seed {checked}."
        )
    return checked


def _with_metadata(row: dict, **metadata) -> dict:
    row.update(metadata)
    return row


def _split_balanced(pool: np.ndarray, assembly: int) -> tuple[np.ndarray, np.ndarray]:
    if 2 * assembly > len(pool):
        raise ValueError("Two index assemblies must fit without overlap.")
    return pool[:assembly].copy(), pool[assembly : 2 * assembly].copy()


def build_bridge(seed: int, config: GateConfig) -> tuple[object, dict]:
    """Build a persistent bridge with a learned neutral-fan-in index relay."""
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
        BrainRegion("cortical_index", config.n_index, **excitatory),
        BrainRegion("cortical_index_fs", config.n_index_fs, **inhibitory),
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
    cfg.enable_coincidence_detection = True
    cfg.coincidence_weighted_drive = True
    cfg.coincidence_k_threshold = float(config.index_coincidence_threshold)
    cfg.coincidence_plateau_strength = float(config.index_plateau_strength)
    cfg.enable_gabab = True
    cfg.gabab_tau_decay = float(config.index_gabab_tau_ms)
    cfg.gabab_propagation_strength = float(config.index_gabab_propagation)

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

    region_names = (
        "ca3",
        "ca1",
        "cortical_cue",
        "cortical_target",
        "cortical_target_fs",
        "cortical_index",
        "cortical_index_fs",
    )
    regions = {
        name: np.asarray(bridge.region_manager.indices(name), dtype=np.int64)
        for name in region_names
    }
    patterns = v1._memory_patterns(seed, config, regions)
    index_a, index_b = _split_balanced(regions["cortical_index"], config.index_assembly)
    patterns["A"]["index"] = index_a
    patterns["B"]["index"] = index_b
    if config.n_target_fs % 2 or config.n_index_fs % 2:
        raise ValueError("Target and index interneuron populations must split evenly.")
    target_fs_a, target_fs_b = np.split(regions["cortical_target_fs"], 2)
    index_fs_a, index_fs_b = np.split(regions["cortical_index_fs"], 2)

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
        "initial_weights": np.full(
            ca3_edges.shape[0], config.ca3_initial_weight, dtype=np.float32,
        ),
        "plastic": True,
        "plasticity_gate": CA3_GATE,
        "conn_type": "E_TO_E",
        "count": int(ca3_edges.shape[0]),
    }
    pairs = {
        "ca3_ca1": [],
        "index_cue": [],
        "target_recurrent": [],
        "target_teacher": [],
        "index_recurrent": [],
        "index_output": [],
    }
    for memory in ("A", "B"):
        pat = patterns[memory]
        pairs["ca3_ca1"].append((pat["ca3"], pat["ca1"]))
        pairs["index_cue"].append((pat["ca1"], pat["cue"]))
        pairs["target_recurrent"].append((pat["target"], pat["target"]))
        pairs["target_teacher"].append((pat["target"], pat["index"]))
        pairs["index_recurrent"].append((pat["index"], pat["index"]))
        pairs["index_output"].append((pat["index"], pat["target"]))

    broad_index = v2._population(
        regions["ca1"],
        regions["cortical_index"],
        config.index_initial_weight,
        plastic=True,
        plasticity_gate=INDEX_TARGET_GATE,
    )
    broad_index["coincidence_detector"] = True
    wiring = {
        "ca3_recurrent": ca3_recurrent,
        "ca3_to_ca1": v2._merge_pairs(
            pairs["ca3_ca1"],
            config.ca3_to_ca1_weight,
            plastic=False,
            plasticity_gate=SCHAFFER_GATE,
            transmission_gate=SCHAFFER_GATE,
        ),
        "ca1_to_cortical_cue": v2._merge_pairs(
            pairs["index_cue"],
            config.index_initial_weight,
            plastic=True,
            plasticity_gate=INDEX_CUE_GATE,
        ),
        "ca1_to_cortical_index": broad_index,
        "target_to_index_teacher": v2._merge_pairs(
            pairs["target_teacher"],
            config.target_to_index_weight,
            plastic=False,
            transmission_gate=WAKE_TEACHING_GATE,
        ),
        "cortical_index_recurrent": v2._merge_pairs(
            pairs["index_recurrent"],
            config.index_recurrent_weight,
            plastic=False,
            plasticity_gate=INDEX_RECURRENT_GATE,
            self_edges=False,
        ),
        "cortical_index_to_target": v2._merge_pairs(
            pairs["index_output"],
            config.index_to_target_weight,
            plastic=False,
            transmission_gate=INDEX_OUTPUT_GATE,
        ),
        "cortical_association": v2._population(
            regions["cortical_cue"],
            regions["cortical_target"],
            config.cortical_initial_weight,
            plastic=True,
            plasticity_gate=CORTICAL_GATE,
        ),
        "cortical_target_recurrent": v2._merge_pairs(
            pairs["target_recurrent"],
            config.cortical_target_recurrent_weight,
            plastic=False,
            plasticity_gate=TARGET_RECURRENT_GATE,
            self_edges=False,
        ),
        "target_to_fs": v2._merge_pairs(
            [
                (patterns["A"]["target"], target_fs_a),
                (patterns["B"]["target"], target_fs_b),
            ],
            config.target_to_fs_weight,
            plastic=False,
            conn_type="E_TO_I",
        ),
        "fs_to_opponent_target": v2._merge_pairs(
            [
                (target_fs_a, patterns["B"]["target"]),
                (target_fs_b, patterns["A"]["target"]),
            ],
            config.fs_to_target_weight,
            plastic=False,
            transmission_gate=TARGET_INHIBITION_GATE,
            conn_type="I_TO_E",
        ),
        "index_to_balance_fs": v2._merge_pairs(
            [
                (patterns["A"]["index"], index_fs_a),
                (patterns["B"]["index"], index_fs_b),
            ],
            config.index_to_fs_weight,
            plastic=False,
            conn_type="E_TO_I",
        ),
        "balance_fs_to_same_index": _with_metadata(
            v2._merge_pairs(
                [
                    (index_fs_a, patterns["A"]["index"]),
                    (index_fs_b, patterns["B"]["index"]),
                ],
                config.fs_to_index_weight,
                plastic=False,
                transmission_gate=INDEX_BALANCE_GATE,
                conn_type="I_TO_E",
            ),
            receptor="gaba_b",
        ),
    }
    inhibitory_indices = np.concatenate(
        [regions["cortical_target_fs"], regions["cortical_index_fs"]]
    )
    bridge.inject_explicit_wiring(
        wiring,
        output_inhibitory_indices=inhibitory_indices.tolist(),
    )
    bridge.set_plasticity_gate(TARGET_RECURRENT_GATE, 0.0)
    bridge.set_plasticity_gate(INDEX_RECURRENT_GATE, 0.0)
    for gate in (
        TARGET_INHIBITION_GATE,
        INDEX_OUTPUT_GATE,
        INDEX_BALANCE_GATE,
        WAKE_TEACHING_GATE,
    ):
        bridge.set_transmission_gate(gate, 1.0)

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
        "inhibitory_indices": inhibitory_indices.copy(),
        "target_fs_pools": {"A": target_fs_a.copy(), "B": target_fs_b.copy()},
        "index_fs_pools": {"A": index_fs_a.copy(), "B": index_fs_b.copy()},
        "neutral_index_fan_in": (
            wiring["ca1_to_cortical_index"]["count"] == config.n_ca1 * config.n_index
        ),
    }
    return bridge, handles


def _set_phase_gates(
    bridge,
    *,
    encode: bool = False,
    sleep: bool = False,
    cortical: bool = False,
) -> None:
    v1._set_phase_gates(bridge, encode=encode, sleep=sleep, cortical=cortical)
    bridge.set_transmission_gate(WAKE_TEACHING_GATE, 1.0 if encode else 0.0)
    bridge.set_transmission_gate(INDEX_OUTPUT_GATE, 1.0 if sleep else 0.0)
    bridge.set_transmission_gate(INDEX_BALANCE_GATE, 1.0 if sleep else 0.0)


def _clear_event_dynamics(bridge, *, preserve_slow_balance: bool) -> None:
    """Reset fast down-state variables; optionally preserve slow local adaptation."""
    v2._clear_fast_dynamics(bridge)
    for name in (
        "cp_conductance_g_coincidence",
        "cp_conductance_g_coincidence_rise",
    ):
        array = getattr(bridge, name, None)
        if array is not None:
            array[:] = 0.0
    if not preserve_slow_balance:
        for name in ("cp_conductance_g_gabab", "cp_conductance_g_gabab_slow"):
            array = getattr(bridge, name, None)
            if array is not None:
                array[:] = 0.0


def _encode_memory(
    bridge,
    handles: dict,
    memory: str,
    events: int,
    config: GateConfig,
) -> dict:
    """Teach the relay synaptically; host current reaches only presented wake populations."""
    _set_phase_gates(bridge, encode=True)
    bridge.set_transmission_gate(TARGET_INHIBITION_GATE, 0.0)
    pat = handles["device_patterns"][memory]
    index_fs = handles["index_fs_pools"][memory]
    spikes = {name: 0 for name in ("ca3", "ca1", "cue", "target", "index", "index_fs")}
    for _ in range(int(events)):
        v1._zero_current(bridge)
        v1._step(bridge, config.encode_off_steps)
        for _ in range(config.encode_on_steps):
            v1._zero_current(bridge)
            for key in ("ca3", "cue", "target"):
                bridge.cp_external_input_current[pat[key]] = config.encode_drive_pA
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
            firing = bridge.cp_firing_states
            for key in ("ca3", "ca1", "cue", "target", "index"):
                spikes[key] += int(firing[pat[key]].sum())
            spikes["index_fs"] += int(firing[index_fs].sum())
    v1._zero_current(bridge)
    return {
        "memory": memory,
        "events": int(events),
        "spikes": spikes,
        "host_driven_populations": ["ca3", "cue", "target"],
        "index_host_driven": False,
    }


def _balance_conductance(bridge, handles: dict) -> dict[str, float]:
    from sim.backend import to_host

    conductance = getattr(bridge, "cp_conductance_g_gabab", None)
    if conductance is None:
        return {"A": 0.0, "B": 0.0}
    host = np.asarray(to_host(conductance), dtype=np.float64)
    return {
        memory: float(host[handles["patterns"][memory]["index"]].mean())
        for memory in ("A", "B")
    }


def _sleep(bridge, handles: dict, condition: str, seed: int, config: GateConfig) -> dict:
    from sim.backend import get_backend

    xp, _ = get_backend()
    # Remove the final wake volley and its short coactivity trace before sleep.
    # Subsequent event boundaries preserve only the slow inter-event balance.
    v1._clear_dynamics(bridge)
    _clear_event_dynamics(bridge, preserve_slow_balance=False)
    cortical_on = condition != "cortical_plasticity_off"
    _set_phase_gates(bridge, sleep=True, cortical=cortical_on)
    bridge.core_config.hebbian_learning_rate = float(config.cortical_sleep_learning_rate)
    if condition == "ca3_ca1_lesion":
        bridge.set_transmission_gate(SCHAFFER_GATE, 0.0)
    target_inhibition_gain = 0.0 if condition == "target_inhibition_lesion" else 1.0
    relay_gain = 0.0 if condition == "index_relay_lesion" else 1.0
    balance_gain = 0.0 if condition == "index_balance_lesion" else 1.0
    bridge.set_transmission_gate(TARGET_INHIBITION_GATE, target_inhibition_gain)
    bridge.set_transmission_gate(INDEX_OUTPUT_GATE, relay_gain)
    bridge.set_transmission_gate(INDEX_BALANCE_GATE, balance_gain)
    shuffled_edges = (
        v1._shuffle_target_index(bridge, seed)
        if condition == "shuffled_target_index"
        else 0
    )

    regions = handles["regions"]
    events = v2._ordered_sleep_events(
        seed,
        config,
        regions["ca3"],
        shuffle=condition == "shuffled_replay_order",
    )
    region_devices = {
        name: xp.asarray(regions[name], dtype=xp.int64)
        for name in (
            "ca3",
            "ca1",
            "cortical_cue",
            "cortical_target",
            "cortical_target_fs",
            "cortical_index",
            "cortical_index_fs",
        )
    }
    ca3_a = handles["device_patterns"]["A"]["ca3"]
    ca3_b = handles["device_patterns"]["B"]["ca3"]
    event_winners: list[str] = []
    spike_totals = {name: 0 for name in region_devices}
    balance_trace: list[dict[str, float]] = []

    for event in events:
        _clear_event_dynamics(bridge, preserve_slow_balance=True)
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
                for name, indices in region_devices.items():
                    spike_totals[name] += int(firing[indices].sum())
        if event_a == event_b == 0:
            event_winners.append("none")
        elif event_a > event_b:
            event_winners.append("A")
        elif event_b > event_a:
            event_winners.append("B")
        else:
            event_winners.append("tie")
        balance_trace.append(_balance_conductance(bridge, handles))

    v1._zero_current(bridge)
    final_balance = _balance_conductance(bridge, handles)
    bridge.set_transmission_gate(TARGET_INHIBITION_GATE, 1.0)
    bridge.set_transmission_gate(INDEX_OUTPUT_GATE, 1.0)
    bridge.set_transmission_gate(INDEX_BALANCE_GATE, 1.0)
    return {
        "events": int(len(events)),
        "event_winners": event_winners,
        "reactivated_events": int(sum(winner != "none" for winner in event_winners)),
        "replayed_A": int(sum(winner == "A" for winner in event_winners)),
        "replayed_B": int(sum(winner == "B" for winner in event_winners)),
        "spikes": spike_totals,
        "shuffled_edges": int(shuffled_edges),
        "event_content_multiset_digest": v2._event_digest(events, order_sensitive=False),
        "event_order_digest": v2._event_digest(events, order_sensitive=True),
        "mean_adjacent_input_overlap": v2._mean_adjacent_overlap(events),
        "target_inhibition_gain_during_sleep": target_inhibition_gain,
        "index_relay_gain_during_sleep": relay_gain,
        "index_balance_gain_during_sleep": balance_gain,
        "index_balance_conductance_final": final_balance,
        "index_balance_conductance_peak": {
            memory: max((row[memory] for row in balance_trace), default=0.0)
            for memory in ("A", "B")
        },
        "host_selected_episode_for_replay": False,
        "host_selected_target_drive": False,
    }


def run_condition(
    seed: int,
    condition: str,
    config: GateConfig | None = None,
    *,
    smoke: bool = False,
) -> dict:
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition {condition!r}; expected one of {CONDITIONS}.")
    if smoke:
        validate_smoke_seed(seed)
    else:
        validate_calibration_seed(seed)
    cfg = config or GateConfig()
    bridge, handles = build_bridge(seed, cfg)
    bridge_ids = [id(bridge)]
    phase_trace: list[str] = []

    bridge.set_transmission_gate(TARGET_INHIBITION_GATE, 0.0)
    before = {
        "ca3": v1._path_weights(bridge, CA3_GATE),
        "index_cue": v1._path_weights(bridge, INDEX_CUE_GATE),
        "index_target": v1._path_weights(bridge, INDEX_TARGET_GATE),
        "cortical": v1._path_weights(bridge, CORTICAL_GATE),
    }
    encode_a = _encode_memory(bridge, handles, "A", cfg.encode_a_events, cfg)
    phase_trace.append("encode_A")
    bridge_ids.append(id(bridge))
    after_a = {
        "ca3": v1._path_weights(bridge, CA3_GATE),
        "cortical": v1._path_weights(bridge, CORTICAL_GATE),
    }
    encode_b = _encode_memory(bridge, handles, "B", cfg.encode_b_events, cfg)
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

    _set_phase_gates(bridge)
    _clear_event_dynamics(bridge, preserve_slow_balance=False)
    recall = {memory: v1._probe_memory(bridge, handles, memory, cfg) for memory in ("A", "B")}
    phase_trace.append("retest")
    bridge_ids.append(id(bridge))

    def mean_delta(later: np.ndarray, earlier: np.ndarray) -> float:
        return float(np.mean(later - earlier))

    return {
        "seed": int(seed),
        "seed_partition": "smoke" if smoke else "calibration",
        "scientific_partition": not smoke,
        "condition": condition,
        "config": asdict(cfg),
        "phase_trace": phase_trace,
        "single_bridge_persisted": len(set(bridge_ids)) == 1,
        "wiring_counts": handles["wiring_counts"],
        "neutral_index_fan_in": handles["neutral_index_fan_in"],
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
            "index_target_during_wake": mean_delta(
                after_b["index_target"], before["index_target"]
            ),
            "index_cue_during_sleep": mean_delta(
                after_sleep["index_cue"], after_b["index_cue"]
            ),
            "index_target_during_sleep": mean_delta(
                after_sleep["index_target"], after_b["index_target"]
            ),
            "cortical_during_wake": mean_delta(after_b["cortical"], before["cortical"]),
            "cortical_during_sleep": mean_delta(after_sleep["cortical"], after_b["cortical"]),
        },
    }


def _mean_recovery(row: dict) -> float:
    return float(np.mean([row["recall"][memory]["correct_rate"] for memory in ("A", "B")]))


def _mean_margin(row: dict) -> float:
    return float(np.mean([row["recall"][memory]["margin"] for memory in ("A", "B")]))


def _mean_false_recall(row: dict) -> float:
    return float(
        np.mean([row["recall"][memory]["false_recall_fraction"] for memory in ("A", "B")])
    )


def _weak_memory_recovery(row: dict) -> float:
    return float(min(row["recall"][memory]["correct_rate"] for memory in ("A", "B")))


def _calibration_verdict(conditions: dict[str, dict]) -> dict:
    intact = conditions["intact"]
    controls = {name: row for name, row in conditions.items() if name != "intact"}
    intact_recovery = _mean_recovery(intact)
    intact_margin = _mean_margin(intact)
    intact_false = _mean_false_recall(intact)
    control_recovery = {name: _mean_recovery(row) for name, row in controls.items()}
    attribution = {
        name: attributable_to(f"v3 replay consolidation versus {name}", intact_recovery, recovery)
        for name, recovery in control_recovery.items()
    }
    temporal = conditions["shuffled_replay_order"]
    plastic_off = conditions["cortical_plasticity_off"]
    inhibition_lesion = conditions["target_inhibition_lesion"]
    relay_lesion = conditions["index_relay_lesion"]
    balance_lesion = conditions["index_balance_lesion"]
    correct_rates = [intact["recall"][memory]["correct_rate"] for memory in ("A", "B")]
    checks = {
        "single_bridge_all_conditions": all(row["single_bridge_persisted"] for row in conditions.values()),
        "both_memories_replayed": intact["sleep"]["replayed_A"] > 0 and intact["sleep"]["replayed_B"] > 0,
        "local_target_inhibition_recruited": intact["sleep"]["spikes"]["cortical_target_fs"] > 0,
        "learned_index_relay_recruited": intact["sleep"]["spikes"]["cortical_index"] > 0,
        "local_slow_balance_recruited": (
            intact["sleep"]["spikes"]["cortical_index_fs"] > 0
            and max(intact["sleep"]["index_balance_conductance_peak"].values()) > 0.0
        ),
        "neutral_index_fan_in": intact["neutral_index_fan_in"],
        "temporal_control_preserves_event_content": (
            intact["sleep"]["event_content_multiset_digest"]
            == temporal["sleep"]["event_content_multiset_digest"]
        ),
        "temporal_control_changes_order": (
            intact["sleep"]["event_order_digest"] != temporal["sleep"]["event_order_digest"]
            and intact["sleep"]["mean_adjacent_input_overlap"]
            > temporal["sleep"]["mean_adjacent_input_overlap"]
        ),
        "index_learns_only_during_wake": (
            intact["weight_deltas"]["index_target_during_wake"] > 1e-5
            and abs(intact["weight_deltas"]["index_target_during_sleep"]) < 1e-7
        ),
        "cortical_weights_change_only_during_sleep": (
            intact["weight_deltas"]["cortical_during_sleep"] > 1e-5
            and abs(intact["weight_deltas"]["cortical_during_wake"]) < 1e-7
            and abs(plastic_off["weight_deltas"]["cortical_during_sleep"]) < 1e-7
        ),
        "intact_partial_recovery": intact_recovery >= 0.03 and intact_margin >= 0.015,
        "both_memories_recovered": all(rate >= 0.015 for rate in correct_rates),
        "weak_memory_balanced": min(correct_rates) >= 0.35 * max(correct_rates),
        "false_recall_bounded": intact_false <= 0.15,
        "intact_beats_no_sleep": intact_recovery >= control_recovery["no_sleep"] + 0.015,
        "intact_beats_shuffled_order": (
            intact_recovery >= control_recovery["shuffled_replay_order"] + 0.01
        ),
        "learned_target_index_beats_shuffle": (
            intact_recovery >= control_recovery["shuffled_target_index"] + 0.015
        ),
        "schaffer_path_is_load_bearing": (
            intact_recovery >= control_recovery["ca3_ca1_lesion"] + 0.015
        ),
        "cortical_plasticity_is_load_bearing": (
            intact_recovery >= control_recovery["cortical_plasticity_off"] + 0.015
        ),
        "index_relay_is_load_bearing": (
            intact_recovery >= control_recovery["index_relay_lesion"] + 0.015
        ),
        "index_balance_improves_weak_memory": (
            _weak_memory_recovery(intact) >= _weak_memory_recovery(balance_lesion) + 0.005
            and intact_false <= _mean_false_recall(balance_lesion) + 0.025
        ),
        "target_inhibition_improves_specificity": (
            intact_false <= _mean_false_recall(inhibition_lesion) - 0.05
            and intact_recovery >= 0.75 * control_recovery["target_inhibition_lesion"]
        ),
    }

    earned = Verdict("replay-driven cortical consolidation v3 calibration")
    earned.require(
        "one bridge persists through every phase and condition",
        checks["single_bridge_all_conditions"],
        expect=True,
    )
    earned.require(
        "all conditions execute the fixed phase sequence",
        all(
            row["phase_trace"] == ["encode_A", "encode_B", "sleep", "retest"]
            for row in conditions.values()
        ),
        expect=True,
    )
    earned.require(
        "wake current never directly drives the cortical index relay",
        all(
            not row[f"encode_{memory}"]["index_host_driven"]
            and row[f"encode_{memory}"]["host_driven_populations"] == ["ca3", "cue", "target"]
            for row in conditions.values()
            for memory in ("A", "B")
        ),
        expect=True,
    )
    earned.require(
        "both wake episodes recruit every required excitatory region",
        all(
            row[f"encode_{memory}"]["spikes"][region] > 0
            for row in conditions.values()
            for memory in ("A", "B")
            for region in ("ca3", "ca1", "cue", "target", "index")
        ),
        expect=True,
    )
    earned.require(
        "intact sleep contains uncued replay, index relay, and both inhibitory loops",
        (
            intact["sleep"]["reactivated_events"] > 0
            and checks["learned_index_relay_recruited"]
            and checks["local_target_inhibition_recruited"]
            and checks["local_slow_balance_recruited"]
        ),
        expect=True,
    )
    earned.require(
        "temporal control changes order while preserving exact event content",
        checks["temporal_control_preserves_event_content"]
        and checks["temporal_control_changes_order"],
        expect=True,
    )
    earned.require(
        "no-sleep control remains quiescent",
        conditions["no_sleep"]["sleep"]["reactivated_events"] == 0
        and sum(conditions["no_sleep"]["sleep"]["spikes"].values()) == 0,
        expect=True,
    )
    earned.require(
        "target inhibition lesion reaches its transmission gate",
        inhibition_lesion["sleep"]["target_inhibition_gain_during_sleep"] == 0.0
        and intact["sleep"]["target_inhibition_gain_during_sleep"] == 1.0,
        expect=True,
    )
    earned.require(
        "index relay lesion reaches its output transmission gate",
        relay_lesion["sleep"]["index_relay_gain_during_sleep"] == 0.0
        and intact["sleep"]["index_relay_gain_during_sleep"] == 1.0,
        expect=True,
    )
    earned.require(
        "index balance lesion reaches its local GABA-B transmission gate",
        balance_lesion["sleep"]["index_balance_gain_during_sleep"] == 0.0
        and intact["sleep"]["index_balance_gain_during_sleep"] == 1.0,
        expect=True,
    )
    earned.require(
        "cortical-plasticity-off holds cortical weights fixed",
        abs(plastic_off["weight_deltas"]["cortical_during_sleep"]) < 1e-7,
        expect=True,
    )
    earned.disabled(
        "STDP, reward modulation, homeostasis, short-term plasticity, and structural plasticity",
        why="bounded isolation of learned relay, local GABA-B adaptation, and Hebbian replay transfer",
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
        "intact_weak_memory_recovery": _weak_memory_recovery(intact),
        "control_mean_recovery": control_recovery,
        "control_mean_false_recall": {
            name: _mean_false_recall(row) for name, row in controls.items()
        },
        "attribution": attribution,
    }


def run_seed(seed: int, config: GateConfig | None = None) -> dict:
    validate_calibration_seed(seed)
    cfg = config or GateConfig()
    conditions = {condition: run_condition(seed, condition, cfg) for condition in CONDITIONS}
    verdict = _calibration_verdict(conditions)
    return {
        "seed": int(seed),
        "seed_partition": "calibration",
        "scientific_partition": True,
        "conditions": conditions,
        "calibration": verdict,
        "calibration_status": verdict["calibration_status"],
    }


def run_calibration(seeds: Iterable[int], config: GateConfig | None = None) -> dict:
    validate_phase("calibration")
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
        "gate": "replay_cortical_consolidation_v3",
        "phase": "calibration",
        "scientific_partition": True,
        "calibration_status": aggregate_status,
        "seeds": list(checked),
        "reserved_seeds_inspected": False,
        "rows": rows,
        "remaining_scaffolds": [
            "host-defined wake episode populations and partial probe cues",
            "fixed cortical relay, teacher, and inhibitory channel membership",
            "host-scheduled sleep down-state boundaries and episode-agnostic CA3 background current",
            "host spike/weight measurement against known calibration assemblies",
            "rate-window Hebbian plasticity and fixed assembly anatomy",
        ],
        "elapsed_seconds": time.time() - started,
    }


def run_smoke(config: GateConfig | None = None) -> dict:
    """Exercise control plumbing without opening a scientific seed or verdict."""
    seed = validate_smoke_seed(SMOKE_SEED)
    cfg = config or smoke_config()
    started = time.time()
    conditions = {
        condition: run_condition(seed, condition, cfg, smoke=True)
        for condition in CONDITIONS
    }
    return {
        "gate": "replay_cortical_consolidation_v3",
        "phase": "smoke",
        "seed": seed,
        "seed_partition": "smoke",
        "scientific_partition": False,
        "calibration_verdict_computed": False,
        "conditions": conditions,
        "structural_checks": {
            "all_conditions_executed": set(conditions) == set(CONDITIONS),
            "fixed_phase_sequence": all(
                row["phase_trace"] == ["encode_A", "encode_B", "sleep", "retest"]
                for row in conditions.values()
            ),
            "single_bridge_persisted": all(
                row["single_bridge_persisted"] for row in conditions.values()
            ),
            "no_scientific_seed_used": all(
                row["seed_partition"] == "smoke"
                and row["scientific_partition"] is False
                for row in conditions.values()
            ),
        },
        "elapsed_seconds": time.time() - started,
    }


def resolve_cli_request(
    *,
    smoke: bool,
    phase: str | None,
    seeds: Iterable[int] | None,
) -> tuple[str, tuple[int, ...]]:
    """Resolve CLI intent without constructing or executing a brain."""
    requested_phase = phase or ("smoke" if smoke else "calibration")
    requested_seeds = None if seeds is None else tuple(int(seed) for seed in seeds)
    if smoke:
        if requested_phase != "smoke":
            raise ValueError("--smoke cannot be combined with --phase calibration")
        checked = (SMOKE_SEED,) if requested_seeds is None else requested_seeds
        if checked != (SMOKE_SEED,):
            raise ValueError(f"--smoke accepts --seeds {SMOKE_SEED} only")
        return "smoke", checked
    if requested_phase == "smoke":
        raise ValueError("--phase smoke requires --smoke")
    validate_phase(requested_phase)
    checked = CALIBRATION_SEEDS if requested_seeds is None else requested_seeds
    return "calibration", validate_calibration_seeds(checked)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("smoke",),
        default=None,
        help="only non-scientific smoke remains selectable; calibration is closed",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="calibration seeds 228/229 are consumed and closed",
    )
    parser.add_argument("--smoke", action="store_true", help="run smoke-only plumbing")
    parser.add_argument("--out", type=Path, default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        requested_phase, seeds = resolve_cli_request(
            smoke=args.smoke,
            phase=args.phase,
            seeds=args.seeds,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if requested_phase == "smoke":
        payload = run_smoke(smoke_config())
    else:
        payload = run_calibration(seeds, GateConfig())
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
