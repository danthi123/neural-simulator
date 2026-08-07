"""Calibration gate for replay-driven cortical consolidation.

The same small spiking bridge persists through four phases:

1. encode episode A;
2. encode an overlapping, interfering episode B;
3. uncued sleep with weak, episode-agnostic CA3 background volleys;
4. retest cortical recall from a partial cue with hippocampal transmission off.

During wake, CA3 recurrence and CA1-to-cortex indexing can learn while the
cortical cue-to-target association is frozen. During sleep those gates swap:
hippocampal weights freeze and replayed CA3/CA1 spikes are the only activity
that can train the cortical association. The no-sleep, shuffled target-index,
CA3-to-CA1-lesion, and cortical-plasticity-off controls isolate that causal
chain. This rung does not yet test temporal replay order.

This is a bounded calibration runner, not a formal gate. Only seeds 212 and
213 are accepted. Seeds 214/215/310 are reserved for development and
311/312/313 for held-out evaluation.

CPU smoke:
    SIM_BACKEND=numpy .venv/bin/python -m research.runners._replay_cortical_consolidation_gate \
        --seeds 212 --smoke

Calibration:
    SIM_BACKEND=numpy .venv/bin/python -m research.runners._replay_cortical_consolidation_gate \
        --seeds 212 213
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from tools.lab import attributable_to
from tools.verdict import UNDEFINED, Verdict

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


CALIBRATION_SEEDS = (212, 213)
DEVELOPMENT_SEEDS = (214, 215, 310)
HELD_OUT_SEEDS = (311, 312, 313)
CONDITIONS = (
    "intact",
    "no_sleep",
    "shuffled_target_index",
    "ca3_ca1_lesion",
    "cortical_plasticity_off",
)

CA3_GATE = "replay_gate_ca3_encode"
INDEX_CUE_GATE = "replay_gate_ca1_cue_index"
INDEX_TARGET_GATE = "replay_gate_ca1_target_index"
CORTICAL_GATE = "replay_gate_cortical_association"
SCHAFFER_GATE = "replay_gate_ca3_to_ca1"


@dataclass(frozen=True)
class GateConfig:
    n_ca3: int = 72
    n_ca1: int = 48
    n_cue: int = 48
    n_target: int = 48
    ca3_assembly: int = 24
    ca3_overlap: int = 0
    ca1_assembly: int = 16
    cue_assembly: int = 16
    cue_overlap: int = 6
    target_assembly: int = 16
    encode_a_events: int = 14
    encode_b_events: int = 20
    encode_on_steps: int = 12
    encode_off_steps: int = 8
    sleep_events: int = 24
    sleep_noise_cells: int = 12
    sleep_noise_steps: int = 15
    sleep_free_steps: int = 15
    probe_steps: int = 45
    probe_cue_fraction: float = 0.50
    encode_drive_pA: float = 1100.0
    sleep_drive_pA: float = 1500.0
    probe_drive_pA: float = 1250.0
    ca3_initial_weight: float = 0.05
    ca3_to_ca1_weight: float = 18.0
    index_initial_weight: float = 0.05
    cortical_initial_weight: float = 0.05
    cortical_target_recurrent_weight: float = 1.0
    hebbian_learning_rate: float = 0.025
    cortical_sleep_learning_rate: float = 2.0
    hebbian_max_weight: float = 90.0
    hebbian_coactivity_decay: float = 0.90
    hebbian_coactivity_thresh: float = 0.01
    propagation_strength: float = 0.12


def smoke_config() -> GateConfig:
    """Small enough for unit tests while exercising every phase."""
    return GateConfig(
        n_ca3=36,
        n_ca1=24,
        n_cue=24,
        n_target=24,
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
        sleep_events=5,
        sleep_noise_cells=5,
        sleep_noise_steps=3,
        sleep_free_steps=6,
        probe_steps=12,
    )


def validate_calibration_seeds(seeds: Iterable[int]) -> tuple[int, ...]:
    checked = tuple(int(seed) for seed in seeds)
    invalid = [seed for seed in checked if seed not in CALIBRATION_SEEDS]
    if invalid:
        raise ValueError(
            f"This bounded runner accepts calibration seeds {CALIBRATION_SEEDS} only; "
            f"refusing reserved seeds {invalid}."
        )
    if not checked:
        raise ValueError("At least one calibration seed is required.")
    return checked


def _all_to_all(pre: np.ndarray, post: np.ndarray, *, self_edges: bool = True) -> tuple[np.ndarray, np.ndarray]:
    edge_pre = np.repeat(pre, len(post))
    edge_post = np.tile(post, len(pre))
    if not self_edges:
        keep = edge_pre != edge_post
        edge_pre, edge_post = edge_pre[keep], edge_post[keep]
    return edge_pre.astype(np.int64), edge_post.astype(np.int64)


def _population(
    pre: np.ndarray,
    post: np.ndarray,
    weight: float,
    *,
    plastic: bool,
    plasticity_gate: str | None = None,
    transmission_gate: str | None = None,
    self_edges: bool = True,
) -> dict:
    edge_pre, edge_post = _all_to_all(pre, post, self_edges=self_edges)
    out = {
        "pre_indices": edge_pre.tolist(),
        "post_indices": edge_post.tolist(),
        "initial_weights": np.full(edge_pre.size, weight, dtype=np.float32),
        "plastic": bool(plastic),
        "conn_type": "E_TO_E",
        "count": int(edge_pre.size),
    }
    if plasticity_gate is not None:
        out["plasticity_gate"] = plasticity_gate
    if transmission_gate is not None:
        out["transmission_gate"] = transmission_gate
    return out


def _memory_patterns(seed: int, config: GateConfig, regions: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    """Create two partially overlapping episodes without using their identity at inference."""
    rng = np.random.default_rng(seed * 31 + 17)

    def overlapping(pool: np.ndarray, size: int, overlap: int) -> tuple[np.ndarray, np.ndarray]:
        draw = rng.choice(pool, 2 * size - overlap, replace=False)
        shared = draw[:overlap]
        a = np.concatenate([shared, draw[overlap:size]])
        b = np.concatenate([shared, draw[size:]])
        return np.sort(a), np.sort(b)

    ca3_a, ca3_b = overlapping(regions["ca3"], config.ca3_assembly, config.ca3_overlap)
    cue_a, cue_b = overlapping(regions["cortical_cue"], config.cue_assembly, config.cue_overlap)
    ca1_draw = rng.choice(regions["ca1"], 2 * config.ca1_assembly, replace=False)
    target_draw = rng.choice(regions["cortical_target"], 2 * config.target_assembly, replace=False)
    ca1_a, ca1_b = np.split(ca1_draw, 2)
    target_a, target_b = np.split(target_draw, 2)
    return {
        "A": {
            "ca3": np.sort(ca3_a),
            "ca1": np.sort(ca1_a),
            "cue": np.sort(cue_a),
            "target": np.sort(target_a),
        },
        "B": {
            "ca3": np.sort(ca3_b),
            "ca1": np.sort(ca1_b),
            "cue": np.sort(cue_b),
            "target": np.sort(target_b),
        },
    }


def build_bridge(seed: int, config: GateConfig) -> tuple[object, dict]:
    """Build the one shared bridge and return its experimental handles."""
    from sim.backend import get_backend
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, GPUConfig, RuntimeState, VisualizationConfig
    from sim.enums import NeuronModel
    from sim.regions import BrainRegion, RegionPathway

    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = [
        BrainRegion("ca3", config.n_ca3, exc_fraction=1.0, internal_density=0.0),
        BrainRegion("ca1", config.n_ca1, exc_fraction=1.0, internal_density=0.0),
        BrainRegion("cortical_cue", config.n_cue, exc_fraction=1.0, internal_density=0.0),
        BrainRegion("cortical_target", config.n_target, exc_fraction=1.0, internal_density=0.0),
    ]
    # A minimal declaration keeps bridge initialization on the region-framework path.
    # It is replaced immediately by the exact experimental wiring below.
    cfg.region_pathways = [
        RegionPathway("ca3", "ca1", density=0.01, weight_mean=0.01, plastic=False),
    ]
    cfg.num_neurons = 0
    cfg.connections_per_neuron = 0
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.dt_ms = 1.0
    cfg.seed = int(seed)
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

    regions = {
        name: np.asarray(bridge.region_manager.indices(name), dtype=np.int64)
        for name in ("ca3", "ca1", "cortical_cue", "cortical_target")
    }
    patterns = _memory_patterns(seed, config, regions)

    # Each episode can form its own recurrent basin. Shared cells still link
    # the memories biologically, but direct A-unique <-> B-unique edges are
    # absent. Deduplicate overlap edges before CSR installation so gate-index
    # offsets remain exact.
    ca3_edges = []
    for memory in ("A", "B"):
        edge_pre, edge_post = _all_to_all(
            patterns[memory]["ca3"], patterns[memory]["ca3"], self_edges=False
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
    for memory in ("A", "B"):
        pat = patterns[memory]
        ca3_to_ca1.append((pat["ca3"], pat["ca1"]))
        index_cue.append((pat["ca1"], pat["cue"]))
        index_target.append((pat["ca1"], pat["target"]))

    def merge_pairs(pairs: list[tuple[np.ndarray, np.ndarray]], weight: float, **kwargs) -> dict:
        populations = [_population(pre, post, weight, **kwargs) for pre, post in pairs]
        return {
            "pre_indices": sum((p["pre_indices"] for p in populations), []),
            "post_indices": sum((p["post_indices"] for p in populations), []),
            "initial_weights": np.concatenate([p["initial_weights"] for p in populations]),
            "plastic": populations[0]["plastic"],
            "plasticity_gate": populations[0].get("plasticity_gate"),
            "transmission_gate": populations[0].get("transmission_gate"),
            "conn_type": "E_TO_E",
            "count": sum(p["count"] for p in populations),
        }

    wiring = {
        "ca3_recurrent": ca3_recurrent,
        "ca3_to_ca1": merge_pairs(
            ca3_to_ca1,
            config.ca3_to_ca1_weight,
            plastic=False,
            plasticity_gate=SCHAFFER_GATE,
            transmission_gate=SCHAFFER_GATE,
        ),
        "ca1_to_cortical_cue": merge_pairs(
            index_cue,
            config.index_initial_weight,
            plastic=True,
            plasticity_gate=INDEX_CUE_GATE,
        ),
        "ca1_to_cortical_target": merge_pairs(
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
        "cortical_target_recurrent": merge_pairs(
            [(patterns["A"]["target"], patterns["A"]["target"]),
             (patterns["B"]["target"], patterns["B"]["target"])],
            config.cortical_target_recurrent_weight,
            plastic=False,
            plasticity_gate="replay_gate_target_recurrent_fixed",
            self_edges=False,
        ),
    }
    bridge.inject_explicit_wiring(wiring)
    bridge.set_plasticity_gate("replay_gate_target_recurrent_fixed", 0.0)
    xp, _ = get_backend()
    handles = {
        "regions": regions,
        "patterns": patterns,
        "device_patterns": {
            memory: {key: xp.asarray(value, dtype=xp.int64) for key, value in pat.items()}
            for memory, pat in patterns.items()
        },
        "wiring_counts": {name: int(pop["count"]) for name, pop in wiring.items()},
        "bridge_identity": id(bridge),
    }
    return bridge, handles


def _step(bridge, steps: int) -> None:
    for _ in range(int(steps)):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1


def _zero_current(bridge) -> None:
    bridge.cp_external_input_current[:] = 0.0


def _set_phase_gates(bridge, *, encode: bool = False, sleep: bool = False, cortical: bool = False) -> None:
    bridge.set_plasticity_gate(CA3_GATE, 1.0 if encode else 0.0)
    bridge.set_plasticity_gate(INDEX_CUE_GATE, 1.0 if encode else 0.0)
    bridge.set_plasticity_gate(INDEX_TARGET_GATE, 1.0 if encode else 0.0)
    bridge.set_plasticity_gate(CORTICAL_GATE, 1.0 if cortical else 0.0)
    bridge.set_plasticity_gate(SCHAFFER_GATE, 0.0)
    bridge.set_transmission_gate(SCHAFFER_GATE, 1.0 if (encode or sleep) else 0.0)


def _clear_dynamics(bridge) -> None:
    """Return activity to rest without changing learned weights."""
    from sim.backend import get_backend

    xp, _ = get_backend()
    cfg = bridge.core_config
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
        arr = getattr(bridge, name, None)
        if arr is not None:
            arr[:] = 0
    trace = getattr(bridge, "cp_hebb_coactivity_trace", None)
    if trace is not None:
        trace[:] = 0.0
    _zero_current(bridge)
    # Keep this assignment backend-neutral and make the intended reset explicit.
    bridge.cp_external_input_current[:] = xp.float32(0.0)
    cfg.current_reward_signal = 0.0


def _path_weights(bridge, gate: str) -> np.ndarray:
    from sim.backend import to_host

    indices = bridge._plasticity_gate_indices_gpu[gate]
    return np.asarray(to_host(bridge.cp_connections.data[indices]), dtype=np.float64)


def _label_free_sparsity(counts: np.ndarray, assembly_size: int) -> dict:
    """Label-free WTA-sparsity / competition-regime statistics of the target pop.

    ADDITIVE INSTRUMENTATION (does not touch any scored value). Computed ONLY from
    the raw per-neuron spike-count vector over ``cortical_target`` and the STRUCTURAL
    assembly SIZE (``config.target_assembly``) -- never the assembly IDENTITY, the
    seed, the correct/wrong labels, or the false-recall metric. This is the STEP-0
    candidate set-point statistic ``S`` for the replay-consolidation self-calibration
    scoping (``_replay_selfcalibration_scoping.md``): does a label-free regime
    statistic predict false-recall across seeds?

    Returns several candidate ``S`` so Step 0 can pick the most monotone:
      * ``pr_eff``  -- participation ratio (Sum c)^2 / Sum(c^2); effective # of active
                       target neurons. LOW  => one-winner (concentrated) regime.
      * ``pr_frac`` -- pr_eff / N in (0,1].
      * ``gini``    -- Gini concentration of counts in [0,1). HIGH => concentrated.
      * ``top_assembly_conc`` -- fraction of all spikes captured by the top
                       ``assembly_size`` neurons (by count). HIGH => one-winner.
      * ``active_fraction`` -- fraction of target neurons that fired at all.
    """
    c = np.asarray(counts, dtype=np.float64).ravel()
    n = int(c.size)
    total = float(c.sum())
    if total <= 0.0 or n == 0:
        return {
            "pr_eff": float("nan"),
            "pr_frac": float("nan"),
            "gini": float("nan"),
            "top_assembly_conc": float("nan"),
            "active_fraction": 0.0,
            "total_spikes": 0.0,
        }
    sum_sq = float((c * c).sum())
    pr_eff = (total * total) / sum_sq if sum_sq > 0.0 else float("nan")
    # Gini concentration (0 = uniform, ->1 = all mass on one unit).
    srt = np.sort(c)
    idx = np.arange(1, n + 1, dtype=np.float64)
    gini = float((np.sum((2.0 * idx - n - 1.0) * srt)) / (n * total))
    k = int(min(max(assembly_size, 1), n))
    top_sorted = np.sort(c)[::-1]
    top_assembly_conc = float(top_sorted[:k].sum() / total)
    active_fraction = float((c > 0).mean())
    return {
        "pr_eff": float(pr_eff),
        "pr_frac": float(pr_eff / n),
        "gini": gini,
        "top_assembly_conc": top_assembly_conc,
        "active_fraction": active_fraction,
        "total_spikes": total,
    }


def _encode_memory(bridge, handles: dict, memory: str, events: int, config: GateConfig) -> dict:
    _set_phase_gates(bridge, encode=True)
    pat = handles["device_patterns"][memory]
    spikes = {name: 0 for name in ("ca3", "ca1", "cue", "target")}
    for _ in range(int(events)):
        _zero_current(bridge)
        _step(bridge, config.encode_off_steps)
        for _ in range(config.encode_on_steps):
            _zero_current(bridge)
            for key in ("ca3", "cue", "target"):
                bridge.cp_external_input_current[pat[key]] = config.encode_drive_pA
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
            firing = bridge.cp_firing_states
            for key in spikes:
                spikes[key] += int(firing[pat[key]].sum())
    _zero_current(bridge)
    return {"memory": memory, "events": int(events), "spikes": spikes}


def _shuffle_target_index(bridge, seed: int) -> int:
    """Destroy CA1-to-target pairing while preserving its exact weight multiset."""
    from sim.backend import get_backend, to_host

    xp, _ = get_backend()
    flat = bridge._plasticity_gate_indices_gpu[INDEX_TARGET_GATE]
    values = np.asarray(to_host(bridge.cp_connections.data[flat])).copy()
    np.random.default_rng(seed * 43 + 5).shuffle(values)
    bridge.cp_connections.data[flat] = xp.asarray(values, dtype=bridge.cp_connections.data.dtype)
    return int(values.size)


def _sleep(bridge, handles: dict, condition: str, seed: int, config: GateConfig) -> dict:
    from sim.backend import get_backend

    xp, _ = get_backend()
    # Sleep starts from quiescence. Carrying the final wake volley into the
    # first sleep steps would let a "no sleep" arm consolidate residual wake
    # activity even though it receives no replay.
    _clear_dynamics(bridge)
    cortical_on = condition != "cortical_plasticity_off"
    _set_phase_gates(bridge, sleep=True, cortical=cortical_on)
    bridge.core_config.hebbian_learning_rate = float(config.cortical_sleep_learning_rate)
    if condition == "ca3_ca1_lesion":
        bridge.set_transmission_gate(SCHAFFER_GATE, 0.0)
    shuffled_edges = _shuffle_target_index(bridge, seed) if condition == "shuffled_target_index" else 0

    regions = handles["regions"]
    ca3_dev = xp.asarray(regions["ca3"], dtype=xp.int64)
    ca1_dev = xp.asarray(regions["ca1"], dtype=xp.int64)
    cue_dev = xp.asarray(regions["cortical_cue"], dtype=xp.int64)
    target_dev = xp.asarray(regions["cortical_target"], dtype=xp.int64)
    ca3_a = handles["device_patterns"]["A"]["ca3"]
    ca3_b = handles["device_patterns"]["B"]["ca3"]
    rng = np.random.default_rng(seed * 59 + 11)
    event_winners: list[str] = []
    spike_totals = {"ca3": 0, "ca1": 0, "cortical_cue": 0, "cortical_target": 0}

    for _event in range(config.sleep_events):
        # Model the cortical/hippocampal down-state between sharp-wave ripple
        # events. Synaptic weights persist; membrane, conductance, and short
        # coactivity state do not leak one replay identity into the next.
        _clear_dynamics(bridge)
        event_a = 0
        event_b = 0
        if condition == "no_sleep":
            _zero_current(bridge)
            for _ in range(config.sleep_noise_steps + config.sleep_free_steps):
                bridge._run_one_simulation_step()
                bridge.runtime_state.current_time_step += 1
        else:
            background = rng.choice(regions["ca3"], config.sleep_noise_cells, replace=False)
            background_dev = xp.asarray(background, dtype=xp.int64)
            for step in range(config.sleep_noise_steps + config.sleep_free_steps):
                _zero_current(bridge)
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
        if event_a == event_b == 0:
            event_winners.append("none")
        else:
            event_winners.append("A" if event_a > event_b else "B" if event_b > event_a else "tie")

    _zero_current(bridge)
    return {
        "events": int(config.sleep_events),
        "event_winners": event_winners,
        "reactivated_events": int(sum(winner != "none" for winner in event_winners)),
        "replayed_A": int(sum(winner == "A" for winner in event_winners)),
        "replayed_B": int(sum(winner == "B" for winner in event_winners)),
        "spikes": spike_totals,
        "shuffled_edges": shuffled_edges,
    }


def _probe_memory(bridge, handles: dict, memory: str, config: GateConfig) -> dict:
    from sim.backend import get_backend, to_host

    xp, _ = get_backend()
    _set_phase_gates(bridge)
    _clear_dynamics(bridge)
    pat = handles["patterns"][memory]
    other = handles["patterns"]["B" if memory == "A" else "A"]
    rng = np.random.default_rng(10_000 + int(bridge.core_config.seed) * 7 + ord(memory))
    cue = pat["cue"].copy()
    rng.shuffle(cue)
    n_partial = max(2, int(round(config.probe_cue_fraction * len(cue))))
    partial = xp.asarray(cue[:n_partial], dtype=xp.int64)
    target_region = handles["regions"]["cortical_target"]
    target_dev = xp.asarray(target_region, dtype=xp.int64)
    counts = np.zeros(len(target_region), dtype=np.float64)
    for _ in range(config.probe_steps):
        _zero_current(bridge)
        bridge.cp_external_input_current[partial] = config.probe_drive_pA
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        counts += np.asarray(to_host(bridge.cp_firing_states[target_dev]), dtype=np.float64)
    _zero_current(bridge)
    local = {int(global_idx): i for i, global_idx in enumerate(target_region)}
    correct_positions = np.asarray([local[int(idx)] for idx in pat["target"]], dtype=np.int64)
    wrong_positions = np.asarray([local[int(idx)] for idx in other["target"]], dtype=np.int64)
    occupied = set(correct_positions.tolist()) | set(wrong_positions.tolist())
    background_positions = np.asarray([i for i in range(len(target_region)) if i not in occupied], dtype=np.int64)
    correct = float(counts[correct_positions].mean() / config.probe_steps)
    wrong = float(counts[wrong_positions].mean() / config.probe_steps)
    background = float(counts[background_positions].mean() / config.probe_steps) if background_positions.size else 0.0
    total = float(counts.sum())
    false_spikes = float(counts[wrong_positions].sum())
    if background_positions.size:
        false_spikes += float(counts[background_positions].sum())
    # ADDITIVE label-free competition-regime statistic S (STEP-0 instrumentation).
    # Computed from the raw counts vector + structural assembly SIZE only -- NO
    # assembly identity / seed / correct-wrong labels / false-recall metric.
    sparsity_S = _label_free_sparsity(counts, int(config.target_assembly))
    return {
        "partial_cue_cells": int(n_partial),
        "correct_rate": correct,
        "wrong_rate": wrong,
        "background_rate": background,
        "margin": correct - max(wrong, background),
        "selectivity": (correct - wrong) / (correct + wrong + 1e-9),
        "false_recall_fraction": false_spikes / total if total > 0.0 else 0.0,
        "total_target_spikes": int(total),
        "sparsity_S": sparsity_S,
    }


def run_condition(seed: int, condition: str, config: GateConfig | None = None) -> dict:
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition {condition!r}; expected one of {CONDITIONS}.")
    validate_calibration_seeds([seed])
    cfg = config or GateConfig()
    bridge, handles = build_bridge(seed, cfg)
    bridge_ids = [id(bridge)]
    phase_trace = []

    before = {
        "ca3": _path_weights(bridge, CA3_GATE),
        "index_cue": _path_weights(bridge, INDEX_CUE_GATE),
        "index_target": _path_weights(bridge, INDEX_TARGET_GATE),
        "cortical": _path_weights(bridge, CORTICAL_GATE),
    }
    encode_a = _encode_memory(bridge, handles, "A", cfg.encode_a_events, cfg)
    phase_trace.append("encode_A")
    bridge_ids.append(id(bridge))
    after_a = {"ca3": _path_weights(bridge, CA3_GATE), "cortical": _path_weights(bridge, CORTICAL_GATE)}
    encode_b = _encode_memory(bridge, handles, "B", cfg.encode_b_events, cfg)
    phase_trace.append("encode_B")
    bridge_ids.append(id(bridge))
    after_b = {
        "ca3": _path_weights(bridge, CA3_GATE),
        "index_cue": _path_weights(bridge, INDEX_CUE_GATE),
        "index_target": _path_weights(bridge, INDEX_TARGET_GATE),
        "cortical": _path_weights(bridge, CORTICAL_GATE),
    }
    sleep = _sleep(bridge, handles, condition, seed, cfg)
    phase_trace.append("sleep")
    bridge_ids.append(id(bridge))
    after_sleep = {
        "ca3": _path_weights(bridge, CA3_GATE),
        "index_cue": _path_weights(bridge, INDEX_CUE_GATE),
        "index_target": _path_weights(bridge, INDEX_TARGET_GATE),
        "cortical": _path_weights(bridge, CORTICAL_GATE),
    }
    recall = {memory: _probe_memory(bridge, handles, memory, cfg) for memory in ("A", "B")}
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


def _calibration_verdict(conditions: dict[str, dict]) -> dict:
    intact = conditions["intact"]
    no_sleep = conditions["no_sleep"]
    shuffled = conditions["shuffled_target_index"]
    lesion = conditions["ca3_ca1_lesion"]
    plastic_off = conditions["cortical_plasticity_off"]

    def mean_recovery(row: dict) -> float:
        return float(np.mean([row["recall"][m]["correct_rate"] for m in ("A", "B")]))

    def mean_margin(row: dict) -> float:
        return float(np.mean([row["recall"][m]["margin"] for m in ("A", "B")]))

    intact_recovery = mean_recovery(intact)
    intact_margin = mean_margin(intact)
    control_recovery = {
        name: mean_recovery(row)
        for name, row in (
            ("no_sleep", no_sleep),
            ("shuffled_target_index", shuffled),
            ("ca3_ca1_lesion", lesion),
            ("cortical_plasticity_off", plastic_off),
        )
    }
    intact_false_recall = float(np.mean([
        intact["recall"][memory]["false_recall_fraction"]
        for memory in ("A", "B")
    ]))
    attribution = {
        name: attributable_to(
            f"replay consolidation versus {name}", intact_recovery, recovery,
        )
        for name, recovery in control_recovery.items()
    }
    checks = {
        "single_bridge_all_conditions": all(row["single_bridge_persisted"] for row in conditions.values()),
        "both_memories_replayed": intact["sleep"]["replayed_A"] > 0 and intact["sleep"]["replayed_B"] > 0,
        "cortical_weights_changed_only_in_intact_sleep": (
            intact["weight_deltas"]["cortical_during_sleep"] > 1e-5
            and abs(intact["weight_deltas"]["cortical_during_wake"]) < 1e-7
            and abs(plastic_off["weight_deltas"]["cortical_during_sleep"]) < 1e-7
        ),
        "intact_partial_recovery": intact_recovery >= 0.03 and intact_margin >= 0.015,
        "both_memories_recovered": all(
            intact["recall"][memory]["correct_rate"] >= 0.01
            for memory in ("A", "B")
        ),
        "false_recall_bounded": intact_false_recall <= 0.15,
        "intact_beats_no_sleep": intact_recovery >= control_recovery["no_sleep"] + 0.015,
        "learned_target_index_beats_shuffle": bool(
            intact_margin >= mean_margin(shuffled) + 0.01
            or np.mean([intact["recall"][m]["false_recall_fraction"] for m in ("A", "B")])
            <= np.mean([shuffled["recall"][m]["false_recall_fraction"] for m in ("A", "B")]) - 0.05
        ),
        "schaffer_path_is_load_bearing": intact_recovery >= control_recovery["ca3_ca1_lesion"] + 0.015,
        "cortical_plasticity_is_load_bearing": intact_recovery >= control_recovery["cortical_plasticity_off"] + 0.015,
    }
    earned = Verdict("replay-driven cortical consolidation calibration")
    earned.require(
        "one bridge persists through every phase and condition",
        all(row["single_bridge_persisted"] for row in conditions.values()),
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
        "both wake episodes recruit hippocampal and cortical populations",
        all(
            row[f"encode_{memory}"]["spikes"][region] > 0
            for row in conditions.values()
            for memory in ("A", "B")
            for region in ("ca3", "ca1", "cue", "target")
        ),
        expect=True,
    )
    earned.require(
        "intact sleep contains uncued replay events",
        intact["sleep"]["reactivated_events"] > 0,
        expect=True,
    )
    earned.require(
        "no-sleep control remains quiescent",
        no_sleep["sleep"]["reactivated_events"] == 0
        and sum(no_sleep["sleep"]["spikes"].values()) == 0,
        expect=True,
    )
    earned.require(
        "cortical-plasticity-off control holds cortical weights fixed",
        abs(plastic_off["weight_deltas"]["cortical_during_sleep"]) < 1e-7,
        expect=True,
    )
    earned.disabled(
        "STDP, reward modulation, homeostasis, short-term plasticity, and structural plasticity",
        why="this rung isolates rate-window Hebbian replay transfer on fixed anatomy",
    )
    decided = earned.decide(go=all(checks.values()), verbose=False)
    return {
        "status": (
            "UNDEFINED"
            if decided["status"] == UNDEFINED
            else "CALIBRATION_PROMISING" if decided["go"] else "CALIBRATION_NOT_YET_CLEAN"
        ),
        "preconditions": decided["preconditions"],
        "disabled_processes": decided["disabled_processes"],
        "undefined_reasons": decided["undefined_reasons"],
        "checks": checks,
        "intact_mean_recovery": intact_recovery,
        "intact_mean_margin": intact_margin,
        "intact_mean_false_recall": intact_false_recall,
        "control_mean_recovery": control_recovery,
        "attributable_fraction": attribution,
    }


def run_seed(seed: int, config: GateConfig | None = None) -> dict:
    validate_calibration_seeds([seed])
    cfg = config or GateConfig()
    rows = {condition: run_condition(seed, condition, cfg) for condition in CONDITIONS}
    return {"seed": int(seed), "conditions": rows, "verdict": _calibration_verdict(rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(CALIBRATION_SEEDS))
    parser.add_argument("--smoke", action="store_true", help="Use the cheap structural/scheduling configuration.")
    parser.add_argument("--out", default=None, help="Optional JSON path. No artifact is written when omitted.")
    args = parser.parse_args()
    seeds = validate_calibration_seeds(args.seeds)
    config = smoke_config() if args.smoke else GateConfig()

    print("[replay-driven cortical consolidation calibration]", flush=True)
    print(f"  backend={os.environ.get('SIM_BACKEND', 'default')} seeds={seeds} smoke={args.smoke}", flush=True)
    print("  phases=encode A -> interfering B -> uncued sleep -> hippocampus-off partial-cue retest", flush=True)
    print(f"  controls={CONDITIONS[1:]}", flush=True)
    started = time.time()
    results = []
    for seed in seeds:
        row = run_seed(seed, config)
        results.append(row)
        verdict = row["verdict"]
        print(
            f"  seed {seed}: {verdict['status']} intact recovery={verdict['intact_mean_recovery']:.4f} "
            f"margin={verdict['intact_mean_margin']:+.4f} controls={verdict['control_mean_recovery']}",
            flush=True,
        )
        print(f"    checks={verdict['checks']}", flush=True)

    payload = {
        "gate": "replay_cortical_consolidation_calibration_v1",
        "seed_policy": {
            "calibration": list(CALIBRATION_SEEDS),
            "development_reserved": list(DEVELOPMENT_SEEDS),
            "held_out_reserved": list(HELD_OUT_SEEDS),
        },
        "config": asdict(config),
        "results": results,
        "elapsed_seconds": time.time() - started,
    }
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"  wrote {out}", flush=True)
    summary = {
        "results": [row["verdict"] for row in results],
        "elapsed_seconds": payload["elapsed_seconds"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
