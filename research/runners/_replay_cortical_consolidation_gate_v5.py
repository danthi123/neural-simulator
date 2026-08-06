"""Bounded v5 calibration: learned, encoding-potentiated CA1->cortex reinstatement.

Root cause banked by the 2026-08-06 research gate: v3/v4 fired the cortical
target during sleep through a FIXED intracortical index->target teacher, so
target reinstatement never depended on the learned, memory-specific hippocampal
index. v1/v2 did carry a plastic ``ca1_to_cortical_target`` pathway but it began
at the same tiny 0.05 efficacy as every other index wire and was seed-fragile:
CA1->cortex reinstatement sat below a reliable operating regime (one calibration
brain learned diffuse false targets, the other barely learned).

V5 makes the CLS reinstatement explicit and reliable (McClelland-McNaughton-
O'Reilly 1995 / Tse 2007): during sharp-wave-ripple replay the hippocampus
REINSTATES the cortical target pattern via CA1->cortex synapses potentiated at
encoding, and repeated co-activation trains the intracortical cue->target
association until it is recallable WITHOUT the hippocampus.

Structural changes vs v2 (which v5 otherwise inherits -- opponent fast-spiking
target competition, target recurrence, the true ``shuffled_replay_order``
control, and the hippocampus-disabled retest):

* the memory-specific ``ca1 -> cortical_target`` pathway starts at a functional
  baseline efficacy (``reinstatement_initial_weight``) and is potentiated by
  wake co-activity (CA1 fires via ca3->ca1 while the target is host-driven), so
  replay reinstates the correct target directly rather than through a fixed
  intracortical teacher;
* that pathway carries its OWN transmission gate (``REINSTATEMENT_GATE``), ON at
  wake-encode and during sleep replay, OFF at the hippocampus-disabled retest, so
  a new ``ca1_target_reinstatement_lesion`` control can silence exactly this wire
  during sleep and prove it is load-bearing for consolidation.

Gating summary:
  wake-encode : ca3->ca1 ON, ca1->target reinstatement ON + plastic, cortical
                association frozen, opponent target-FS OFF (host drives target);
  sleep       : ca3->ca1 ON, reinstatement ON (transmission), cortical
                association plastic, opponent target-FS ON;
  retest      : ca3->ca1 OFF (CA1 silent), reinstatement OFF -> recall must come
                from the consolidated intracortical cue->target association.

The named-but-parked surpass (2026-08-06 gate): if the reinstated target latches
to a single global winner under capacity pressure, the deliverable names
spike-frequency-adaptation-driven one-of-N eviction on the target attractor. It
is NOT built here unless that pathology appears; no conductance calibration is in
scope.

Remaining scaffolds are explicit and unchanged from v2: host-defined wake
episode populations and partial probe cues; fixed opponent/inhibitory channel
membership; host-scheduled sleep down-state boundaries and episode-agnostic CA3
background current; host spike/weight measurement against known assemblies; the
rate-window Hebbian rule and fixed assembly anatomy as simplified biological
stand-ins.

Fresh seed partition (disjoint from v1-v4): calibration 412/413, smoke 416,
development 414/415/410 and held-out 417/418/419 remain mechanically rejected
until calibration lands a clean verdict.

CPU smoke:
    SIM_BACKEND=numpy .venv/bin/python -m research.runners._replay_cortical_consolidation_gate_v5 --smoke

Calibration:
    SIM_BACKEND=numpy .venv/bin/python -m research.runners._replay_cortical_consolidation_gate_v5 \
        --seeds 412 413 --out research/findings/raw/replay_v5/replay_v5_calibration.json
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


CALIBRATION_SEEDS = (412, 413)
DEVELOPMENT_SEEDS = (414, 415, 410)
HELD_OUT_SEEDS = (417, 418, 419)
SMOKE_SEED = 416
CONDITIONS = (
    "intact",
    "no_sleep",
    "shuffled_replay_order",
    "shuffled_target_index",
    "ca3_ca1_lesion",
    "cortical_plasticity_off",
    "target_inhibition_lesion",
    "ca1_target_reinstatement_lesion",
)

CA3_GATE = v1.CA3_GATE
INDEX_CUE_GATE = v1.INDEX_CUE_GATE
INDEX_TARGET_GATE = v1.INDEX_TARGET_GATE
CORTICAL_GATE = v1.CORTICAL_GATE
SCHAFFER_GATE = v1.SCHAFFER_GATE
TARGET_INHIBITION_GATE = v2.TARGET_INHIBITION_GATE
TARGET_RECURRENT_GATE = v2.TARGET_RECURRENT_GATE
REINSTATEMENT_GATE = "replay_v5_ca1_to_target_reinstatement"


@dataclass(frozen=True)
class GateConfig(v2.GateConfig):
    """V5 inherits v2 anatomy/timing; adds an explicit CA1->cortex reinstatement wire."""

    # Functional baseline efficacy of the memory-specific CA1->cortical_target
    # reinstatement pathway (v1/v2 started this at index_initial_weight=0.05 and
    # never reached a reliable reinstatement). Wake co-activity potentiates it
    # further via the INDEX_TARGET_GATE Hebbian gate.
    reinstatement_initial_weight: float = 8.0
    # The cortical CUE must also be reinstated during replay so it co-activates
    # with the reinstated target and the intracortical cue->target association
    # can consolidate (CLS reinstates the WHOLE cortical memory, not just one
    # pole). v1/v2 left ca1->cue at 0.05, so the cue barely fired during sleep
    # and the association could not form. Wake co-activity potentiates this too.
    cue_reinstatement_initial_weight: float = 8.0
    # Transmission gain applied to the reinstatement wire during sleep replay.
    reinstatement_sleep_gain: float = 1.0


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
    invalid = [
        seed
        for seed in checked
        if seed not in CALIBRATION_SEEDS and seed != SMOKE_SEED
    ]
    if invalid:
        raise ValueError(
            f"This bounded v5 runner accepts calibration seeds {CALIBRATION_SEEDS} "
            f"(or smoke seed {SMOKE_SEED}) only; refusing reserved seeds {invalid}."
        )
    if not checked:
        raise ValueError("At least one calibration seed is required.")
    return checked


def _set_phase_gates(
    bridge, *, encode: bool = False, sleep: bool = False, cortical: bool = False
) -> None:
    """v1 phase gates plus the explicit CA1->cortex reinstatement transmission gate."""
    v1._set_phase_gates(bridge, encode=encode, sleep=sleep, cortical=cortical)
    bridge.set_transmission_gate(REINSTATEMENT_GATE, 1.0 if (encode or sleep) else 0.0)


def build_bridge(seed: int, config: GateConfig) -> tuple[object, dict]:
    """One bridge with a strong, memory-specific, gated CA1->cortical_target wire."""
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
    reinstatement = []
    recurrent_target = []
    for memory in ("A", "B"):
        pat = patterns[memory]
        ca3_to_ca1.append((pat["ca3"], pat["ca1"]))
        index_cue.append((pat["ca1"], pat["cue"]))
        reinstatement.append((pat["ca1"], pat["target"]))
        recurrent_target.append((pat["target"], pat["target"]))

    wiring = {
        "ca3_recurrent": ca3_recurrent,
        "ca3_to_ca1": v2._merge_pairs(
            ca3_to_ca1,
            config.ca3_to_ca1_weight,
            plastic=False,
            plasticity_gate=SCHAFFER_GATE,
            transmission_gate=SCHAFFER_GATE,
        ),
        "ca1_to_cortical_cue": v2._merge_pairs(
            index_cue,
            config.cue_reinstatement_initial_weight,
            plastic=True,
            plasticity_gate=INDEX_CUE_GATE,
        ),
        # The load-bearing v5 wire: a memory-specific CA1->cortical_target
        # reinstatement pathway at a functional baseline efficacy, plastic during
        # wake encode (INDEX_TARGET_GATE) and separately transmission-gated
        # (REINSTATEMENT_GATE) so sleep can drive it and retest can silence it.
        "ca1_to_cortical_target": v2._merge_pairs(
            reinstatement,
            config.reinstatement_initial_weight,
            plastic=True,
            plasticity_gate=INDEX_TARGET_GATE,
            transmission_gate=REINSTATEMENT_GATE,
        ),
        "cortical_association": v2._population(
            regions["cortical_cue"],
            regions["cortical_target"],
            config.cortical_initial_weight,
            plastic=True,
            plasticity_gate=CORTICAL_GATE,
        ),
        "cortical_target_recurrent": v2._merge_pairs(
            recurrent_target,
            config.cortical_target_recurrent_weight,
            plastic=False,
            plasticity_gate=TARGET_RECURRENT_GATE,
            self_edges=False,
        ),
        "target_to_fs": v2._merge_pairs(
            [
                (patterns["A"]["target"], fs_a),
                (patterns["B"]["target"], fs_b),
            ],
            config.target_to_fs_weight,
            plastic=False,
            conn_type="E_TO_I",
        ),
        "ca1_to_target_fs": v2._merge_pairs(
            [
                (patterns["A"]["ca1"], fs_a),
                (patterns["B"]["ca1"], fs_b),
            ],
            config.ca1_to_fs_weight,
            plastic=False,
            conn_type="E_TO_I",
        ),
        "fs_to_target": v2._merge_pairs(
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
    bridge.set_transmission_gate(REINSTATEMENT_GATE, 1.0)

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
        "reinstatement_memory_specific": all(
            wiring["ca1_to_cortical_target"]["count"]
            == sum(len(patterns[m]["ca1"]) * len(patterns[m]["target"]) for m in ("A", "B"))
            for _ in (0,)
        ),
    }
    return bridge, handles


def _encode_memory(bridge, handles: dict, memory: str, events: int, config: GateConfig) -> dict:
    """Wake encode with reinstatement transmission ON so CA1->target co-activity potentiates."""
    _set_phase_gates(bridge, encode=True)
    return v1._encode_memory(bridge, handles, memory, events, config)


def _sleep(bridge, handles: dict, condition: str, seed: int, config: GateConfig) -> dict:
    from sim.backend import get_backend

    xp, _ = get_backend()
    v1._clear_dynamics(bridge)
    cortical_on = condition != "cortical_plasticity_off"
    _set_phase_gates(bridge, sleep=True, cortical=cortical_on)
    bridge.core_config.hebbian_learning_rate = float(config.cortical_sleep_learning_rate)
    if condition == "ca3_ca1_lesion":
        bridge.set_transmission_gate(SCHAFFER_GATE, 0.0)
    inhibition_gain = 0.0 if condition == "target_inhibition_lesion" else 1.0
    reinstatement_gain = (
        0.0
        if condition == "ca1_target_reinstatement_lesion"
        else float(config.reinstatement_sleep_gain)
    )
    bridge.set_transmission_gate(TARGET_INHIBITION_GATE, inhibition_gain)
    bridge.set_transmission_gate(REINSTATEMENT_GATE, reinstatement_gain)
    shuffled_edges = (
        v1._shuffle_target_index(bridge, seed)
        if condition == "shuffled_target_index"
        else 0
    )

    regions = handles["regions"]
    events = v2._ordered_sleep_events(
        seed, config, regions["ca3"], shuffle=condition == "shuffled_replay_order",
    )
    ca3_dev = xp.asarray(regions["ca3"], dtype=xp.int64)
    ca1_dev = xp.asarray(regions["ca1"], dtype=xp.int64)
    cue_dev = xp.asarray(regions["cortical_cue"], dtype=xp.int64)
    target_dev = xp.asarray(regions["cortical_target"], dtype=xp.int64)
    fs_dev = xp.asarray(regions["cortical_target_fs"], dtype=xp.int64)
    ca3_a = handles["device_patterns"]["A"]["ca3"]
    ca3_b = handles["device_patterns"]["B"]["ca3"]
    target_a = handles["device_patterns"]["A"]["target"]
    target_b = handles["device_patterns"]["B"]["target"]
    event_winners: list[str] = []
    target_winners: list[str] = []
    spike_totals = {
        "ca3": 0,
        "ca1": 0,
        "cortical_cue": 0,
        "cortical_target": 0,
        "cortical_target_fs": 0,
    }

    for event in events:
        v2._clear_fast_dynamics(bridge)
        event_a = event_b = 0
        target_ev_a = target_ev_b = 0
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
                target_ev_a += int(firing[target_a].sum())
                target_ev_b += int(firing[target_b].sum())
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
        # Did the CORTICAL target that was reinstated match the CA3 assembly that
        # replayed? A memory-specific reinstatement co-activates the matching
        # target; a diffuse global winner would not track the replayed episode.
        if target_ev_a == target_ev_b == 0:
            target_winners.append("none")
        elif target_ev_a > target_ev_b:
            target_winners.append("A")
        elif target_ev_b > target_ev_a:
            target_winners.append("B")
        else:
            target_winners.append("tie")

    v1._zero_current(bridge)
    bridge.set_transmission_gate(TARGET_INHIBITION_GATE, 1.0)
    bridge.set_transmission_gate(REINSTATEMENT_GATE, 1.0)
    reinstatement_match = sum(
        1
        for ca3_win, tgt_win in zip(event_winners, target_winners)
        if ca3_win in ("A", "B") and tgt_win == ca3_win
    )
    reinstatement_mismatch = sum(
        1
        for ca3_win, tgt_win in zip(event_winners, target_winners)
        if ca3_win in ("A", "B") and tgt_win in ("A", "B") and tgt_win != ca3_win
    )
    return {
        "events": int(len(events)),
        "event_winners": event_winners,
        "target_winners": target_winners,
        "reactivated_events": int(sum(winner != "none" for winner in event_winners)),
        "replayed_A": int(sum(winner == "A" for winner in event_winners)),
        "replayed_B": int(sum(winner == "B" for winner in event_winners)),
        "target_reinstated_A": int(sum(winner == "A" for winner in target_winners)),
        "target_reinstated_B": int(sum(winner == "B" for winner in target_winners)),
        "reinstatement_match_events": int(reinstatement_match),
        "reinstatement_mismatch_events": int(reinstatement_mismatch),
        "spikes": spike_totals,
        "shuffled_edges": int(shuffled_edges),
        "event_content_multiset_digest": v2._event_digest(events, order_sensitive=False),
        "event_order_digest": v2._event_digest(events, order_sensitive=True),
        "mean_adjacent_input_overlap": v2._mean_adjacent_overlap(events),
        "target_inhibition_gain_during_sleep": inhibition_gain,
        "reinstatement_gain_during_sleep": reinstatement_gain,
        "host_selected_episode_for_replay": False,
        "host_selected_target_drive": False,
    }


def run_condition(seed: int, condition: str, config: GateConfig | None = None) -> dict:
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition {condition!r}; expected one of {CONDITIONS}.")
    validate_calibration_seeds([seed])
    cfg = config or GateConfig()
    bridge, handles = build_bridge(seed, cfg)
    bridge_ids = [id(bridge)]
    phase_trace: list[str] = []

    # Wake teaching first: opponent target inhibition must not suppress the
    # externally presented target while the hippocampal index is being learned.
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

    # Retest with hippocampus disabled: CA1 silent (Schaffer off) and the
    # reinstatement wire explicitly off, so recall can only come from the
    # consolidated intracortical cue->target association.
    _set_phase_gates(bridge)
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
        "reinstatement_memory_specific": bool(handles["reinstatement_memory_specific"]),
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
            "reinstatement_during_wake": mean_delta(
                after_b["index_target"], before["index_target"]
            ),
            "index_cue_during_sleep": mean_delta(after_sleep["index_cue"], after_b["index_cue"]),
            "reinstatement_during_sleep": mean_delta(
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
    return float(np.mean([row["recall"][memory]["false_recall_fraction"] for memory in ("A", "B")]))


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
        name: attributable_to(f"v5 replay consolidation versus {name}", intact_recovery, recovery)
        for name, recovery in control_recovery.items()
    }
    temporal = conditions["shuffled_replay_order"]
    plastic_off = conditions["cortical_plasticity_off"]
    inhibition_lesion = conditions["target_inhibition_lesion"]
    reinstatement_lesion = conditions["ca1_target_reinstatement_lesion"]
    correct_rates = [intact["recall"][memory]["correct_rate"] for memory in ("A", "B")]

    checks = {
        "single_bridge_all_conditions": all(
            row["single_bridge_persisted"] for row in conditions.values()
        ),
        "reinstatement_memory_specific": all(
            row["reinstatement_memory_specific"] for row in conditions.values()
        ),
        "both_memories_replayed": intact["sleep"]["replayed_A"] > 0
        and intact["sleep"]["replayed_B"] > 0,
        "target_reinstated_during_sleep": intact["sleep"]["spikes"]["cortical_target"] > 0
        and intact["sleep"]["reinstatement_match_events"] > 0,
        "reinstatement_tracks_replayed_episode": (
            intact["sleep"]["reinstatement_match_events"]
            > intact["sleep"]["reinstatement_mismatch_events"]
        ),
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
        "reinstatement_potentiates_during_wake": (
            intact["weight_deltas"]["reinstatement_during_wake"] > 1e-5
        ),
        "intact_partial_recovery": intact_recovery >= 0.03 and intact_margin >= 0.015,
        "both_memories_recovered": all(rate >= 0.015 for rate in correct_rates),
        "weak_memory_present": min(correct_rates) >= 0.25 * max(correct_rates + [1e-9]),
        "false_recall_bounded": intact_false <= 0.15,
        # THE load-bearing consolidation test: the reinstated cortical trace
        # survives the hippocampus-disabled retest ONLY when replay ran.
        "intact_beats_no_sleep": intact_recovery >= control_recovery["no_sleep"] + 0.015,
        "intact_beats_shuffled_order": (
            intact_recovery >= control_recovery["shuffled_replay_order"] + 0.01
        ),
        "schaffer_path_is_load_bearing": (
            intact_recovery >= control_recovery["ca3_ca1_lesion"] + 0.015
        ),
        "cortical_plasticity_is_load_bearing": (
            intact_recovery >= control_recovery["cortical_plasticity_off"] + 0.015
        ),
        # The new v5 wire is causally necessary for consolidation.
        "reinstatement_is_load_bearing": (
            intact_recovery >= control_recovery["ca1_target_reinstatement_lesion"] + 0.015
        ),
        "target_inhibition_improves_specificity": (
            intact_false <= _mean_false_recall(inhibition_lesion) - 0.05
            and intact_recovery >= 0.75 * control_recovery["target_inhibition_lesion"]
        ),
    }

    earned = Verdict("replay-driven cortical consolidation v5 calibration")
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
        "the CA1->target reinstatement wire is memory-specific by anatomy",
        checks["reinstatement_memory_specific"],
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
        "intact sleep reinstates the correct target and recruits opponent FS",
        (
            intact["sleep"]["reactivated_events"] > 0
            and checks["target_reinstated_during_sleep"]
            and checks["local_inhibition_recruited"]
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
    earned.reaches(
        "reinstatement lesion silences the CA1->target wire during sleep",
        before=intact["sleep"]["reinstatement_gain_during_sleep"],
        after=reinstatement_lesion["sleep"]["reinstatement_gain_during_sleep"],
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
    earned.floor(
        "intact hippocampus-independent recovery vs no-replay floor",
        intact_recovery,
        control_recovery["no_sleep"],
    )
    earned.control(
        "intact vs CA1->target reinstatement lesion",
        treatment=intact_recovery,
        control=control_recovery["ca1_target_reinstatement_lesion"],
    )
    earned.disabled(
        "STDP, reward modulation, homeostasis, short-term plasticity, and structural plasticity",
        why="bounded isolation of learned CA1->cortex reinstatement + Hebbian replay transfer",
    )
    decided = earned.decide(go=all(checks.values()), verbose=False)
    return {
        "calibration_status": (
            "UNDEFINED"
            if decided["status"] == UNDEFINED
            else "CALIBRATION_PROMISING" if decided["go"] else "CALIBRATION_NEEDS_REVISION"
        ),
        "verdict": decided["status"],
        "preconditions": decided["preconditions"],
        "disabled_processes": decided["disabled_processes"],
        "undefined_reasons": decided["undefined_reasons"],
        "checks": checks,
        "intact_mean_recovery": intact_recovery,
        "intact_mean_margin": intact_margin,
        "intact_mean_false_recall": intact_false,
        "intact_weak_memory_recovery": _weak_memory_recovery(intact),
        "intact_reinstatement_match_events": intact["sleep"]["reinstatement_match_events"],
        "intact_reinstatement_mismatch_events": intact["sleep"]["reinstatement_mismatch_events"],
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
        "verdict": verdict["verdict"],
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
        "gate": "replay_cortical_consolidation_v5",
        "phase": "calibration",
        "mechanism": "learned encoding-potentiated CA1->cortical_target reinstatement",
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
    if args.smoke:
        seeds = (SMOKE_SEED,)
        config = smoke_config()
    else:
        seeds = args.seeds
        config = GateConfig()
    payload = run_calibration(seeds, config)
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
